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

# 双边的oncompletion中，当wc status是flush error的时候，return，真的是正确的吗？

果然不对，现在这样才是正确的，事实上我们需要取消on_conncetion的注释，这可能是作者留下的小关卡，看你是不是真的理解透彻了。

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

# 步骤

## 状态转换

Client: SS_INIT → SS_MR_SENT → SS_RDMA_SENT → SS_DONE_SENT
Server: SS_INIT → SS_MR_SENT → SS_RDMA_SENT → SS_DONE_SENT

Both:   RS_INIT → RS_MR_RECV → RS_DONE_RECV

## 具体步骤
1. 初始化，包括build_connection, build_context, register_memory（注册MR，用于msg双向写）
2. 

## RDMA Buffer
假设是client向server写的模式
1. 在client中准备buffer：sprintf(get_local_message_region(id->context), "message from active/client side with pid %d",getpid());
2. 


## 处理竞态

情况1：Server先收到Client的MR
Server: recv_state=RS_INIT → RS_MR_RECV (收到Client的MR)
Server: send_state=SS_INIT (还没发送自己的MR)
→ 触发 send_mr(conn)，Server发送自己的MR

情况2：Client先发送，Server后发送
Server: send_state=SS_INIT → SS_MR_SENT (已经发送了自己的MR)
Server: recv_state=RS_INIT → RS_MR_RECV (收到Client的MR)  
→ 不触发 send_mr()，避免重复发送



## 数据流

Client ←→ Server: MSG_MR (双边，交换内存信息)
Client  → Server: RDMA WRITE (单边，传输实际数据)
Client ←→ Server: MSG_DONE (双边，确认完成)