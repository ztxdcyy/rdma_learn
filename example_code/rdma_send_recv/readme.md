## Quick Start: ##

### Server: ###
    - make  
    - ./server  

### Client: ###
    - make
    - ./client <server address> <server port(default: 22222)>

### ENV

我们需要安装两个RDMA相关开发包：

sudo apt-get install librdmacm-dev libibverbs-dev

建立软连接：

ln -sf /usr/lib/x86_64-linux-gnu/libibverbs.so.1 /usr/lib/x86_64-linux-gnu/libibverbs.so
ln -sf /usr/lib/x86_64-linux-gnu/librdmacm.so.1 /usr/lib/x86_64-linux-gnu/librdmacm.so

给ibs11绑定ip：
sudo ip addr add 192.168.100.1/24 dev ibs11
sudo ip link set ibs11 up
ip addr show ibs11

### Learn Note

- 你用的 rdma_cm（如例程 client/server.c）是“基于IP的连接管理方式”，即通过 IP 地址+端口号来自动发现、路由和建立 RDMA QP。它本质上是 RDMA 规范中“Connection Manager”层的设计，目的是让 RDMA 编程像 TCP 一样简单（但底层数据流依然是 RDMA verbs，不走传统IP栈）。
- 不是所有 RDMA 应用都需要 IB 端口分配 IP，但只要用 rdma_cm 相关 API（如 rdma_create_id/rdma_resolve_addr），就必须给 IB 端口分配 IP，否则无法连接。
- 如果你用的是“裸 verbs 编程”模式（如 ibv_create_qp + 交换 lid/qpn/pkey 等），则不需要分配 IP，可直接用 lid/gid 做通信参数。
- 所以，只要你跑的是 rdma_cm 相关例程，配置 IB 端口 IP 是必需的，这不是混淆，而是厂商 API 层设计的结果。

所以我们后续可以进一步使用纯ibv编程而不是基于tcp/ip之上

## 现在啥情况？
client: 

```
root@k8s-node07:/sgl-workspace/nvshmem_learn/example_code/rdma_send_recv# ./client 192.168.100.1 22222
address resolved.
route resolved.
connected. posting send...
send completed successfully.
on_completion: status is not IBV_WC_SUCCESS.
```
client send成功，但是没有拿到serve返回的一个啥玩意

server:

```
root@k8s-node07:/sgl-workspace/nvshmem_learn/example_code/rdma_send_recv# ./server
listening on port 22222.
received connection request.
received message: message from active/client side with pid 1380595
peer disconnected.
```
server 能接收到东西，报了一个“peer disconnected”

根据send这个双向的设计，问题应该出在【client拿不到“server接收成功的事件”】上。

具体点，ibv_poll_cq之后，会调用on_completion，然后在这个函数里返回报错 IBV_WC_SUCCESS

## IBV_WC_STATUS 不成功的原因

client 正常 post_recv，并且只投递了一次 recv buffer。只要 server 不回发数据，client 的唯一一次 post_recv 不会被使用，不会出错。
但 server 断开连接时，RDMA 协议会 flush 所有未完成的 WR，这时 client 的 CQ 可能收到 IBV_WC_WR_FLUSH_ERR 或类似 error CQE，这就是你看到 on_completion: status is not IBV_WC_SUCCESS 的根本原因。
这不是逻辑bug，而是 RDMA verbs 规范的表现。只要 QP 被销毁，所有未完成的 WR（如未真正被 remote send 消耗的 recv）都会 error flush。
