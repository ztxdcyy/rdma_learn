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