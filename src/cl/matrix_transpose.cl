#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6
#define TILE_SIZE 16

__kernel void matrix_transpose(__global const float* a,
                               __global       float* b,
                               unsigned int M,
                               unsigned int K)
{
    const unsigned int indexX = get_global_id(0);
    const unsigned int indexY = get_global_id(1);

    const unsigned int local_indexX = get_local_id(0);
    const unsigned int local_indexY = get_local_id(1);

    __local float title[TILE_SIZE][TILE_SIZE];
    if (indexY < M && indexX < K) {
        title[local_indexX][local_indexY] = a[indexY * K + indexX];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int begX = (indexX/TILE_SIZE)*TILE_SIZE;
    unsigned int begY = (indexY/TILE_SIZE)*TILE_SIZE;

    if (begX + local_indexY < K && begY + local_indexX < M) {
        b[(begX + local_indexY) * M + begY + local_indexX] = title[local_indexY][local_indexX];
    }
}