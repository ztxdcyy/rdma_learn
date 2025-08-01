## Quick Start: ##

### Server: ###
    - make  
    - ./rdma-server <write/read>  

### Client: ###
    - make
    - ./rdma-client <write/read> <server address> <server port(default: 22222)>

## 现在啥问题？

```
./rdma-client write 10.179.114.197 19780
address resolved.
route resolved.
send completed successfully.
received MSG_MR. writing message to remote memory...
on_completion: status is not IBV_WC_SUCCESS.

./rdma-server read
listening on port 19780.
received connection request.
send completed successfully.
received MSG_MR. reading message from remote memory...
on_completion: status is not IBV_WC_SUCCESS.
```

应该是要打印出pid的，但是没看到啊
在on_completion里面添加上了打印status code以及具体status string的代码后发现：

```
./rdma-client write 10.179.114.197 21606
address resolved.
route resolved.
wc->status = 0 (success)
send completed successfully.
wc->status = 0 (success)
received MSG_MR. writing message to remote memory...
wc->status = 5 (Work Request Flushed Error)
on_completion: status is not IBV_WC_SUCCESS.
```

其实这样是正确的，单边就应该是这样的，但是退出的不是很优雅？

比如我们设置client write，而server read，
那么我们应该在server read完之后在server触发打印，还是在client write之后通过完成信号触发打印？还是其他？

我们的writer究竟会写什么呢？
我如何在reader上验证单边写是否真的做完了呢？
我可以通过写入地址和buffer长度做打印吗（需要从metadata中读取？）？

> client RDMA write 完成后发 MSG_DONE，server 收到后打印自己的 buffer。


# 双边的oncompletion中，当wc status是flush error的时候，return，真的是正确的吗？