## Quick Start: ##

### Server: ###
    - make  
    - ./rdma-server <write/read>  

### Client: ###
    - make
    - ./rdma-client <write/read> <server address> <server port(default: 22222)>


# 单边 write 模式 log
Client:
```
root@k8s-node07:/sgl-workspace/nvshmem_learn/example_code/rdma_write_read# ./rdma-client write 10.179.114.197 60196
address resolved.
route resolved.
[on_completion] wr_id=0x5629769acf00, status=0 (success), opcode=0, send_state=0, recv_state=0
send completed successfully.
[on_completion] wr_id=0x5629769acf00, status=0 (success), opcode=128, send_state=1, recv_state=0
received MSG_MR. writing message to remote memory...
[DEBUG] About to post RDMA WRITE
[DEBUG] peer_mr.addr=0x5638f0daab10, peer_mr.rkey=0x1ff7b5
[DEBUG] local_region=0x5629769b0050, lkey=0x1ffcba
[on_completion] wr_id=0x5629769acf00, status=0 (success), opcode=1, send_state=1, recv_state=1
send completed successfully.
[on_completion] wr_id=0x5629769acf00, status=0 (success), opcode=0, send_state=2, recv_state=1
send completed successfully.
[on_completion] wr_id=0x5629769acf00, status=0 (success), opcode=128, send_state=3, recv_state=1
[LOG] WRITE: Both DONE, calling rdma_disconnect(conn->id)
remote buffer: message from passive/server side with pid 873
disconnected.
[LOG] destroy_connection called for conn=0x5629769acf00, id=0x5629769ab670
```

Server:
```
</heter-compute/example_code/rdma_write_read# ./rdma-server write                      
listening on port 60196.
received connection request.
[on_completion] wr_id=0x5638f0daaa70, status=0 (success), opcode=128, send_state=0, recv_state=0
[on_completion] wr_id=0x5638f0daaa70, status=0 (success), opcode=0, send_state=0, recv_state=1
send completed successfully.
received MSG_MR. writing message to remote memory...
[DEBUG] About to post RDMA WRITE
[DEBUG] peer_mr.addr=0x5629769b0460, peer_mr.rkey=0x1ffdbb
[DEBUG] local_region=0x5638f0da8630, lkey=0x1ff5b3
[on_completion] wr_id=0x5638f0daaa70, status=0 (success), opcode=128, send_state=1, recv_state=1
[on_completion] wr_id=0x5638f0daaa70, status=0 (success), opcode=1, send_state=1, recv_state=2
send completed successfully.
[on_completion] wr_id=0x5638f0daaa70, status=0 (success), opcode=0, send_state=2, recv_state=2
send completed successfully.
[LOG] WRITE: Both DONE, calling rdma_disconnect(conn->id)
remote buffer: message from active/client side with pid 3494716
peer disconnected.
[LOG] destroy_connection called for conn=0x5638f0daaa70, id=0x5638f0da8140
```

# 是否注释on_connection有什么区别？

发生在server.c的on_event中，之前是r=0，注释掉了r=on_connection，也就是不做回应，此时client直接被flush掉。注释之后会发送一条收到的ACK，此时client会收到server的一条消息。

这是取消注释前后的对比：

Client对比：

```
root@k8s-node07:/sgl-workspace/nvshmem_learn/example_code/rdma_send_recv# ./client 10.179.114.197 22222
address resolved.
route resolved.
connected. posting send...
wc->status = 0 (success)
send completed successfully.
wc->status = 5 (Work Request Flushed Error)
disconnected.

root@k8s-node07:/sgl-workspace/nvshmem_learn/example_code/rdma_send_recv# ./client 10.179.114.197 22222
address resolved.
route resolved.
connected. posting send...
wc->status = 0 (success)
send completed successfully.
wc->status = 0 (success)
received message: message from passive/server side with pid 1093734
disconnected.
```

Server 对比：

```
[root@k8s-node06 rdma_send_recv]# ./server 
listening on port 22222.
received connection request.
received message: 你好，我在测试RDMA SEND DEMO，可以看到我的消息吗？
peer disconnected.
^C

[root@k8s-node06 rdma_send_recv]# ./server 
listening on port 22222.
received connection request.
received message: 你好，我在测试RDMA SEND DEMO，可以看到我的消息吗？
connected. posting send...
send completed successfully.
peer disconnected.
^C
```

可以看到server发出了"connected. posting send...""send completed successfully"，而且client收到了message from server

所以之前碰到flush就直接return的做法显然是不正确的。

而之前为什么会遇到flush？主要是server这边on_connection被注释掉了,event过来之后，不能正确的进入on—connection case。所以server主动断开连接了，导致qp flush，导致client 收到flush error。

```
int on_event(struct rdma_cm_event *event)
{
  int r = 0;

  if (event->event == RDMA_CM_EVENT_CONNECT_REQUEST)      // 当server收到连接请求事件后
    r = on_connect_request(event->id);
  else if (event->event == RDMA_CM_EVENT_ESTABLISHED)     // 当server收到建立事件后
    r = on_connection(event->id->context);                // server会向client发送一条欢迎（在这里是pid）意思就是确立连接了。（但是不写也可以的，这和TCP的三次握手的ack不一样）
    // r = 0;                                             // 啊我明白了，作者意思其实就是说，这个onconnection就是不重要的，有没有都行，不影响server收到client发过来的消息。
  else if (event->event == RDMA_CM_EVENT_DISCONNECTED)    // 当server收到断开事件后
    r = on_disconnect(event->id);                         // 执行销毁的后处理，包括释放qp、mr资源，断开本次的rdma连接等等
  else
    die("on_event: unknown event.");

  return r;
}
```

# 数据结构

## connection
```
struct connection {
  struct rdma_cm_id *id;
  struct ibv_qp *qp;    // queue pair

  int connected;

  // mr memory region 锁页 防止被操作系统换出
  struct ibv_mr *recv_mr;
  struct ibv_mr *send_mr;
  struct ibv_mr *rdma_local_mr;
  struct ibv_mr *rdma_remote_mr;

  struct ibv_mr peer_mr;

  struct message *recv_msg;
  struct message *send_msg;

  char *rdma_local_region;
  char *rdma_remote_region;

  enum {
    SS_INIT,
    SS_MR_SENT,
    SS_RDMA_SENT,
    SS_DONE_SENT
  } send_state;

  enum {
    RS_INIT,
    RS_MR_RECV,
    RS_DONE_RECV
  } recv_state;
};
```

简单来说有
- 四种mr，收发普通mr，收发rdma mr
- 两种msg 收发
- 两种状态，收发

## message
```
struct message {
  enum {
    MSG_MR,
    MSG_DONE
  } type;

  union {
    struct ibv_mr mr;
  } data;
};
```
message有两种
1. MSG_MR消息：
- 包含对方的Memory Region信息
- 包括远程内存地址和访问key
- 用于建立RDMA操作的目标
2. MSG_DONE消息：
- 通知对方RDMA操作已完成
- 用于同步和协调

发送方: send_msg (MSG_MR) → 网络 → 接收方: recv_msg
发送方: send_msg (MSG_DONE) → 网络 → 接收方: recv_msg

下面将说明，为什么MSG_MR中包含了mr addr length 和 rkey，之后的数据流向是如何的？

## 什么是post？
软件下发任务给硬件在RDMA中称为post
- 对于接收方来说，软件告知硬件，接收到的数据放在哪里，称为post_recv
- 对于发送方来说，软件告知硬件，准备发送addr+len的数据，称为post_send

具体我们看一下void post_receives(struct connection *conn) 中需要准备哪些东西？

```
void post_receives(struct connection *conn)
{
  struct ibv_recv_wr wr, *bad_wr = NULL;
  struct ibv_sge sge;

  wr.wr_id = (uintptr_t)conn;
  wr.next = NULL;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  sge.addr = (uintptr_t)conn->recv_msg;
  sge.length = sizeof(struct message);
  sge.lkey = conn->recv_mr->lkey;

  TEST_NZ(ibv_post_recv(conn->qp, &wr, &bad_wr));
}
```

可以看到最核心的内容就是sge，里面包含了首地址addr，内存块长度length，lkey

- local key：MR注册好之后会返回LKEY和RKEY，LKEY用于自己访问自己，RKEY用于别人访问自己。一片内存区可以多次注册MR，每次可以设置不同的访问权限，每次都会返回不同的LKEY和RKEY。
- MR注册来自于：ibv_reg_mr() 注册与Protection Domain关联的内存区域 (MR)。通过这样做，允许 RDMA 设备向该内存读取和写入数据。执行此注册需要一些时间，因此当需要快速响应时，不建议在数据路径中执行内存注册。



# 解析

## 具体步骤
1. 初始化，包括build_connection, build_context, register_memory（注册MR，用于msg双向写）
2. 

## RDMA Buffer
假设是client向server写的模式
1. 在client中准备buffer：sprintf(get_local_message_region(id->context), "message from active/client side with pid %d",getpid());
2. 



## 数据流

Client ←→ Server: MSG_MR (双边，交换内存信息)

Client.rdma_local_region → (单边RDMA Write) → Server.rdma_remote_region
Server.rdma_local_region → (单边RDMA Write) → Client.rdma_remote_region

Client ←→ Server: MSG_DONE (双边，确认完成)

## 具体步骤和收发前后

Write模式下：

Client端:
- rdma_local_region: "message from active/client side with pid XXX"  ← 数据准备在这里
- rdma_remote_region: [空]

Server端:
- rdma_local_region: "message from passive/server side with pid XXX"  ← 数据准备在这里
- rdma_remote_region: [空] → 等待接收Client的RDMA Write

1. write模式下，get-local-message-region返回rdma local region
2. client调用get-local-message-region，得到rdma local region，在这段buffer内写入想要传递的信息
3. server端也会往自己的rdma-local-region中写入信息sprintf
4. write模式下，client和server执行rdma write
5. 执行完rdma write之后，client和server的rdma region如下：

数据流向：

Client ←→ Server: MSG_MR (双边，交换内存信息)

Client.rdma_local_region → (单边RDMA Write) → Server.rdma_remote_region
Server.rdma_local_region → (单边RDMA Write) → Client.rdma_remote_region

Client ←→ Server: MSG_DONE (双边，确认完成)

Client端:
- rdma_local_region: "message from active/client side with pid XXX"  （原始数据）
- rdma_remote_region: "message from passive/server side with pid XXX" （接收到server的数据）

Server端:
- rdma_local_region: "message from passive/server side with pid XXX"  （原始数据）
- rdma_remote_region: "message from active/client side with pid XXX" (接收到Client数据)

【write按照read的形式组织一下】
---
Read模式下；
数据准备：
```
Client端内存：
├─ rdma_local_region:  [空白]                                          ← 接收缓冲区
└─ rdma_remote_region: "message from active/client side with pid XXX"  ← 准备给对方读取

Server端内存：
├─ rdma_local_region:  [空白]                                          ← 接收缓冲区  
└─ rdma_remote_region: "message from passive/server side with pid XXX" ← 准备给对方读取
```
数据流向：
```
Client ←→ Server: MSG_MR (双边，交换内存信息)

Client.rdma_local_region <- (单边RDMA Read) <- Server.rdma_remote_region
Server.rdma_local_region <- (单边RDMA Read) <- Client.rdma_remote_region

Client ←→ Server: MSG_DONE (双边，确认完成)
```

完成read后：
```
Client端内存：
├─ rdma_local_region:  "message from passive/server side with pid XXX"  ← 从Server读取
└─ rdma_remote_region: "message from active/client side with pid XXX"   ← 原始准备数据

Server端内存：
├─ rdma_local_region:  "message from active/client side with pid XXX"   ← 从Client读取
└─ rdma_remote_region: "message from passive/server side with pid XXX"  ← 原始准备数据
```
## 状态转换

Client: SS_INIT → SS_MR_SENT → SS_RDMA_SENT → SS_DONE_SENT
Server: SS_INIT → SS_MR_SENT → SS_RDMA_SENT → SS_DONE_SENT

Both:   RS_INIT → RS_MR_RECV → RS_DONE_RECV

触发RDMA操作的条件：

```
if (conn->send_state == SS_MR_SENT && conn->recv_state == RS_MR_RECV) {
    // 只有当这个条件满足时，才执行RDMA操作
    // 执行IBV_WR_RDMA_WRITE
}
```

接下来我们分析一下，状态变化的时序逻辑

初始状态：
```
Client: send_state=SS_INIT, recv_state=RS_INIT
Server: send_state=SS_INIT, recv_state=RS_INIT
```
连接建立后：

Step 1 - Client主动发送MR：
```
// rdma-client.c 的 on_connection()
send_mr(id->context);  // Client主动发送自己的MR信息
```
Step 2 - Client状态变化：
```
Client: send_state=SS_MR_SENT, recv_state=RS_INIT
```
Step 3 - Server接收到Client的MR：
```
// 在on_completion中
if (conn->recv_msg->type == MSG_MR) {
    conn->recv_state++;  // Server: recv_state变为RS_MR_RECV
    
    if (conn->send_state == SS_INIT)  // Server还没发送自己的MR：on_completion在server和client都会被执行，但是这里的if只有server会触发，因为这里client的ss已经是ss-mr-sent，而不是init了。具体来说，on_completion来自于pollcq轮询cq进程
        send_mr(conn);  // Server发送自己的MR
}
```
Step 4 - Server状态变化：
```
Server: send_state=SS_MR_SENT, recv_state=RS_MR_RECV  ← 条件满足！
```
Server先满足，此时会先执行rdma write

Step 5 - Client接收到Server的MR：
```
Client: send_state=SS_MR_SENT, recv_state=RS_MR_RECV  ← 条件也满足！
```

