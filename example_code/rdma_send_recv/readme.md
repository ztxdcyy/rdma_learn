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

### Learn Note

- 你用的 rdma_cm（如例程 client/server.c）是“基于IP的连接管理方式”，即通过 IP 地址+端口号来自动发现、路由和建立 RDMA QP。它本质上是 RDMA 规范中“Connection Manager”层的设计，目的是让 RDMA 编程像 TCP 一样简单（但底层数据流依然是 RDMA verbs，不走传统IP栈）。
- 不是所有 RDMA 应用都需要 IB 端口分配 IP，但只要用 rdma_cm 相关 API（如 rdma_create_id/rdma_resolve_addr），就必须给 IB 端口分配 IP，否则无法连接。
- 如果你用的是“裸 verbs 编程”模式（如 ibv_create_qp + 交换 lid/qpn/pkey 等），则不需要分配 IP，可直接用 lid/gid 做通信参数。
- 所以，只要你跑的是 rdma_cm 相关例程，配置 IB 端口 IP 是必需的，这不是混淆，而是厂商 API 层设计的结果。

所以我们后续可以进一步使用纯ibv编程而不是基于tcp/ip之上



