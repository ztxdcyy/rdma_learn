#include <stdio.h>
#include "mpi.h"
#include "nvshmem.h"
#include "nvshmemx.h"

// CUDA CHECK 错误检查宏
#define CUDA_CHECK(stmt)                                  \
do {                                                      \
    cudaError_t result = (stmt);                          \
    if (cudaSuccess != result) {                          \
        fprintf(stderr, "[%s:%d] CUDA failed with %s \n", \
         __FILE__, __LINE__, cudaGetErrorString(result)); \
        exit(-1);                                         \
    }                                                     \
} while (0)

__global__ void simple_shift(int *destination) {
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int peer = (mype + 1) % npes;

    // 单边通信操作，直接写入对称内存：https://docs.nvidia.com/nvshmem/api/gen/api/rma.html#nvshmem-p
    // destination: 即将写入的对称内存中的地址；mype：需要写入的值； peer: the number of the remote PE
    nvshmem_int_p(destination, mype, peer);
}

int main (int argc, char *argv[]) {
    // ========= 1. 参数以及结构体初始化 ===============
    int mype_node, msg;
    cudaStream_t stream;        // 工程实践一般都会开一个stream，即使脚本只用了一个stream。对应后面的async kernel
    int rank, nranks;
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;         // attr 初始化参数结构体，初始化为默认值，后续可以设置UID
    nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;             // 调用 NVSHMEMX_UNIQUEID_INITIALIZER 初始化 UID 结构体

    // ========= 2. MPI进程初始化 ================
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);           // 从整体通信域 MPI_COMM_WORLD 获取本进程的进程号，写入rank对应地址，直接修改值（cpp基础太不扎实了太垃圾了）
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);         // 获取总进程数，同理，传指针，函数直接把结果写到地址里，这样就修改了变量的值
    
    // ======== 3. NVSHMEM通信域初始化（微信面对面建群，号码来自于UID）==============
    // 由 某一个进程（比如PE0）拿到一个UID 防止多个不同的 id导致冲突，比如就是一堆人建群，只要一个人提供微信面对面建群的号码就行，人多了大家不知道听谁的。
    if (rank == 0) {
        nvshmemx_get_uniqueid(&id);
        // 打印unique id的前16字节（通常足够观察，实际大小依赖实现）
        unsigned char* id_bytes = (unsigned char*)&id;
        printf("rank 0 unique id: ");
        for (size_t i = 0; i < sizeof(nvshmemx_uniqueid_t); ++i) {
            printf("%02x", id_bytes[i]);
            if ((i+1) % 4 == 0) printf(" "); // 分组美观
        }
        printf("\n");
    }

    // PE 0 broadcast the unique ID to all peers
    // NVSHMEM需要所有进程用同一个“唯一ID”来初始化对称空间，确保这些进程属于同一个通信域，否则各自的对称空间无法互通。
    // 获得了微信面对面建群号码之后，告诉所有人（广播），大家都输入这个号码，就能进同一个群啦（初始化nv通信域）
    MPI_Bcast(&id, sizeof(nvshmemx_uniqueid_t), MPI_UINT8_T, 0, MPI_COMM_WORLD);
    // attr是一个包含了各种参数，包括uid，rank，nranks，id等参数的结构体。这里意思是写入
    nvshmemx_set_attr_uniqueid_args(rank, nranks, &id, &attr);
    // nvshmem通过这个attr进行初始化，第一个参数指定是UID方式，第二个参数传入我们刚定义好的attr
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
    // 获取当前PE在整个team里的rank，这里的team指的是除了全局PE之外的组织形式，比如全局四张卡，我可以指定两卡组成一个team
    // NVSHMEMX_TEAM_WORLD 表示包含所有PE的全局team，NVSHMEMX_TEAM_NODE 表示当前node内的所有PE组成的Team
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    // 根据rank号 绑定cuda device
    CUDA_CHECK(cudaSetDevice(mype_node));
    // 创建cuda stream
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ======= 4.分配对称内存+启动kernel ===========
    // 在对称内存上分配内存，nvshmem malloc 返回int指针变量
    int *destination = (int *) nvshmem_malloc (sizeof(int));
    // 在一个stream上异步启动kernel，执行nvshmem单边写操作
    simple_shift<<<1, 1, 0, stream>>>(destination);
    // nvshmem阻塞：卡内同步cudaStreamSynchronize(streams)+分布式同步，也就是等待mpi网络内所有PE
    nvshmemx_barrier_all_on_stream(stream);


    // 异步内存拷贝，把gpu上dest地址里东西拷贝到主机cpu上msg指针（总之是传递指针，指针对起来就行）方便后续printf
    // cudaError_t cudaMemcpyAsync(     // cudaError_t 执行状态
    //     void *dst,                // 目标地址（主机或设备）
    //     const void *src,          // 源地址（主机或设备）
    //     size_t count,             // 拷贝字节数
    //     cudaMemcpyKind kind,      // 拷贝类型（HostToDevice, DeviceToHost等）
    //     cudaStream_t stream       // 指定的CUDA stream
    // );

    
    // ======= 5. 从device拷贝结果到host的msg，打印 ===============
    // 这里的dest发生了语义转换：在nvshmem_int_p里，确实指的是写入到对称内存上，也就是目标地址。但是在下面的cudaMemcpyAsync里，dest反而意味着src，也就是从dest的对称内存中拷贝信息到&msg
    CUDA_CHECK(cudaMemcpyAsync(&msg, destination, sizeof(int),
                cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    printf("%d: received message %d\n", nvshmem_my_pe(), msg);

    nvshmem_free(destination);
    nvshmem_finalize();
    MPI_Finalize();
    return 0;
}