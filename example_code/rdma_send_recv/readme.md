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

## 基本概念
- context结构体由若干结构体组成，包括ibv_context上下文（ibv是ib的用户态的接口，ib verbs。除了用户态还有内核态，内核态接口就是ib_xxx）
- PD：protected region 保护区域，防止被操作系统swap这部分物理内存。换句话说，就是绑定了物理地址和虚拟地址
Work Queue(WQ): 工作队列，在发送过程中 WQ =  SQ; 在接收过程中WQ = WQ;
- CQ：Complete Queue，完成队列，WQ完成之后会生成一个完成事件CQE，先进先出的存在CQ里面。之后，CQ用于告诉用户WQ上的消息已经被处理完成；
- Completion Channel：完成通道，用于让内核/驱动通过事件通知（event notification）的方式告知用户空间“完成队列（CQ）有新事件”，代替了传统的轮询模式。
- cq_poller_thread：CQ轮询线程，独立线程负责监听 comp_channel 上的事件通知，并在有事件时调用 ibv_poll_cq 获取并处理所有“完成项（Work Completion, WC）”。

- WR全称为Work Request，意为工作请求；WC全称Work Completion，意为工作完成。这两者其实是WQE和CQE在用户层的“映射”。因为APP是通过调用协议栈接口来完成RDMA通信的，WQE和CQE本身并不对用户可见，是驱动中的概念。用户真正通过API下发的是WR，收到的是WC。


参考：[知乎专栏 _ RDMA基本元素](https://zhuanlan.zhihu.com/p/141267386)

## 两种建立连接的方式

建立连接的方式主要有两种，一种是通过Socket连接来交换信息，跟普通的网络通信过程没有区别；另一种是通过CM协议，通过保留的QP1来交换后续通信所需的信息。交换必要的信息之后，两端的应用才会开始准备WQE。CM建链在IB规范第12章有讲，以后我也会专门介绍。


## Send / Receive 的具体建立双向链接的步骤：
client（send） -> "message from active/client side with pid %d" -> server（recv）

client（recv）<-  "message from passive/server side with pid %d"   <- server（send）

1. 接收端APP以WQE的形式下发一次RECV任务到RQ。on_connect_request[post_receives[ibv_post_recv]]
2. 发送端APP以WQE的形式下发一次SEND任务到SQ。on_connection[ibv_post_send]
3. 发送端硬件从SQ中拿到任务书，从内存中拿到待发送数据，组装数据包。
4. 发送端网卡将数据包通过物理链路发送给接收端网卡。
5. 接收端收到数据，进行校验后回复ACK报文给发送端。
6. 接收端硬件从RQ中取出一个任务书（WQE）。main[on_event[on_connect_request[build_context[poll_cq[ibv_poll_cq]]]]]
7. 接收端硬件将数据放到WQE中指定的位置，然后生成“任务报告”CQE，放置到CQ中。 poll_cq
8. 接收端APP取得任务完成信息。 printf
9. 发送端网卡收到ACK后，生成CQE，放置到CQ中。
10. 发送端APP取得任务完成信息。

poll就是轮询：以某一间隔持续主动查询某东西的状态，每一秒来问问你咋样了