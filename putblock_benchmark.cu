/*
 * 性能测试版本的 put-block.cu
 * 添加了 CUDA Event 计时功能
 */

#include <stdio.h>
#include <assert.h>

#include "bootstrap_helper.h"
#include "nvshmem.h"
#include "nvshmemx.h"

#undef CUDA_CHECK
#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            exit(-1);                                                             \
        }                                                                         \
    } while (0)

#define THREADS_PER_BLOCK 1024
#define NUM_WARMUP_RUNS 5    // 预热运行次数
#define NUM_TIMING_RUNS 10   // 正式计时运行次数

__global__ void set_and_shift_kernel(float *send_data, float *recv_data, int num_elems, int mype,
                                     int npes) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (thread_idx < num_elems) send_data[thread_idx] = mype;

    int peer = (mype + 1) % npes;
    int block_offset = blockIdx.x * blockDim.x;
    nvshmemx_float_put_block(recv_data + block_offset, send_data + block_offset,
                             min(blockDim.x, num_elems - block_offset),
                             peer);
}

// 性能测试函数
void benchmark_put_block(float *send_data, float *recv_data, int num_elems, int mype, int npes, int num_blocks) {
    cudaEvent_t start, stop;
    float total_time = 0.0f;
    
    // 创建 CUDA Event
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    printf("[PE %d] Starting performance benchmark...\n", mype);
    
    // 预热运行 - 不计时
    printf("[PE %d] Warming up (%d runs)...\n", mype, NUM_WARMUP_RUNS);
    for (int i = 0; i < NUM_WARMUP_RUNS; i++) {
        set_and_shift_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(send_data, recv_data, num_elems, mype, npes);
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all(); // 确保所有PE同步
    }
    
    // 正式计时运行
    printf("[PE %d] Timing runs (%d iterations)...\n", mype, NUM_TIMING_RUNS);
    
    for (int run = 0; run < NUM_TIMING_RUNS; run++) {
        // 重置数据
        CUDA_CHECK(cudaMemset(send_data, 0, num_elems * sizeof(float)));
        CUDA_CHECK(cudaMemset(recv_data, 0, num_elems * sizeof(float)));
        
        nvshmem_barrier_all(); // 确保所有PE准备就绪
        
        // 开始计时
        CUDA_CHECK(cudaEventRecord(start));
        
        // 执行传输
        set_and_shift_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(send_data, recv_data, num_elems, mype, npes);
        
        // 结束计时
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        // 等待所有PE完成
        nvshmem_barrier_all();
        
        // 计算时间
        float elapsed_time;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        total_time += elapsed_time;
        
        printf("[PE %d] Run %d: %.3f ms\n", mype, run + 1, elapsed_time);
    }
    
    // 计算性能指标
    float avg_time = total_time / NUM_TIMING_RUNS;
    float data_size_mb = (num_elems * sizeof(float)) / (1024.0f * 1024.0f);
    float bandwidth_gbps = (data_size_mb / (avg_time / 1000.0f)) / 1024.0f;
    
    printf("\n=== PE %d Performance Results ===\n", mype);
    printf("Data size: %.2f MB (%d elements)\n", data_size_mb, num_elems);
    printf("Average time: %.3f ms\n", avg_time);
    printf("Bandwidth: %.2f GB/s\n", bandwidth_gbps);
    printf("Latency: %.3f μs\n", avg_time * 1000.0f);
    printf("=====================================\n\n");
    
    // 清理
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main(int c, char *v[]) {
    int mype, npes, mype_node;
    float *send_data, *recv_data;
    int num_elems = 8192;
    int num_blocks;

#ifdef NVSHMEMTEST_MPI_SUPPORT
    bool use_mpi = false;
    char *value = getenv("NVSHMEMTEST_USE_MPI_LAUNCHER");
    if (value) use_mpi = atoi(value);
#endif

#ifdef NVSHMEMTEST_MPI_SUPPORT
    if (use_mpi) {
        nvshmemi_init_mpi(&c, &v);
    } else
        nvshmem_init();
#else
    nvshmem_init();
#endif

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

    CUDA_CHECK(cudaSetDevice(mype_node));
    send_data = (float *)nvshmem_malloc(sizeof(float) * num_elems);
    recv_data = (float *)nvshmem_malloc(sizeof(float) * num_elems);
    assert(send_data != NULL && recv_data != NULL);

    assert(num_elems % THREADS_PER_BLOCK == 0);
    num_blocks = num_elems / THREADS_PER_BLOCK;

    // 执行性能测试
    benchmark_put_block(send_data, recv_data, num_elems, mype, npes, num_blocks);

    // 功能验证 (保留原有逻辑)
    printf("[PE %d] Running correctness test...\n", mype);
    
    // 重置并执行一次传输用于验证
    CUDA_CHECK(cudaMemset(send_data, 0, num_elems * sizeof(float)));
    CUDA_CHECK(cudaMemset(recv_data, 0, num_elems * sizeof(float)));
    
    set_and_shift_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(send_data, recv_data, num_elems, mype, npes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    // 数据验证
    float *host = new float[num_elems];
    CUDA_CHECK(cudaMemcpy(host, recv_data, num_elems * sizeof(float), cudaMemcpyDefault));
    int ref = (mype - 1 + npes) % npes;
    bool success = true;
    for (int i = 0; i < num_elems; ++i) {
        if (host[i] != ref) {
            printf("Error at %d of rank %d: %f (expected %d)\n", i, mype, host[i], ref);
            success = false;
            break;
        }
    }

    if (success) {
        printf("[%d of %d] Correctness test PASSED\n", mype, npes);
    } else {
        printf("[%d of %d] Correctness test FAILED\n", mype, npes);
    }

    delete[] host;
    nvshmem_free(send_data);
    nvshmem_free(recv_data);
    nvshmem_finalize();

#ifdef NVSHMEMTEST_MPI_SUPPORT
    if (use_mpi) nvshmemi_finalize_mpi();
#endif
    return 0;
}