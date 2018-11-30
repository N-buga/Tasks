#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6
#define TILE_SIZE 4

__kernel void matrix_multiplication(__global const float* a,
                                    __global const float* b,
                                    __global       float* c,
                                     unsigned int M,
                                     unsigned int K,
                                     unsigned int N)
{
    const unsigned int indexX = get_global_id(0);
    const unsigned int indexY = get_global_id(1);

    const unsigned int local_indexX = get_local_id(0);
    const unsigned int local_indexY = get_local_id(1);

    float sum = 0;

    for (int i = 0; i < K; i += TILE_SIZE) {
        __local float tile1[TILE_SIZE][TILE_SIZE];
        for (int q = 0; q < TILE_SIZE; q++) {
            if (indexX < N && indexY < M && (i + q) < K) {
                tile1[local_indexY][q] = a[indexY * K + i + q];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        __local float tile2[TILE_SIZE][TILE_SIZE];
        for (int q = 0; q < TILE_SIZE; q++) {
            if (indexX < N && indexY < M && (i + q) < K) {
                tile2[q][local_indexX] = b[(i + q) * N + indexX];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int q = 0; q < TILE_SIZE; q++) {
            if (i + q < K) {
                sum += tile1[local_indexY][q] * tile2[q][local_indexX];
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (indexY < M && indexX < N) {
        c[indexY * N + indexX] = sum;
    }
}