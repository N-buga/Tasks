#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 128

void swap_global(__global float* as, unsigned int i, unsigned int j) {
    float val = as[i];
    as[i] = as[j];
    as[j] = val;
}

void swap_local(__local float* as, unsigned int i, unsigned int j) {
    float val = as[i];
    as[i] = as[j];
    as[j] = val;
}

__kernel void bitonic_local(__global float* as,
                             unsigned int n,
                             unsigned int i,
                             unsigned int beg_j)
{
    const unsigned int index = get_global_id(0);
    const unsigned int local_index = get_local_id(0);

    __local float as_local[WORK_GROUP_SIZE];

    if (index < n) {
        as_local[local_index] = as[index];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int j = beg_j; j >= 1; j /= 2) {
        if (index % (2 * j) < j && index + j < n) {
            if (index % (4 * i) < 2*i) {
                if (as_local[local_index] > as_local[local_index + j]) {
                    swap_local(as_local, local_index, local_index + j);
                }
            } else {
                if (as_local[local_index] < as_local[local_index + j]) {
                    swap_local(as_local, local_index, local_index + j);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (index < n) {
        as[index] = as_local[local_index];
    }
}

__kernel void bitonic_global(__global float* as,
                      unsigned int n,
                      unsigned int i,
                      unsigned int j)
{
    const unsigned int index = get_global_id(0);

    if (index % (2 * j) < j && index + j < n) {
        if (index % (4 * i) < 2*i) {
            if (as[index] > as[index + j]) {
                swap_global(as, index, index + j);
            }
        } else {
            if (as[index] < as[index + j]) {
                swap_global(as, index, index + j);
            }
        }
    }
}
