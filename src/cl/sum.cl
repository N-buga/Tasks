#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 128
__kernel void sum(__global const unsigned int* as,
                  unsigned int n,
                  __global unsigned int* res)
{
    const unsigned int global_id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);

    __local unsigned int local_as[WORK_GROUP_SIZE];
    local_as[local_id] = as[global_id];

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int arr_size = WORK_GROUP_SIZE;
    while (arr_size >= 2) {
        if (local_id < arr_size/2) {
            local_as[local_id] += local_as[arr_size/2 + local_id];
        }
        if (arr_size/2*2 != arr_size && local_id == 0) {
            local_as[0] += local_as[arr_size/2 + 1];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        arr_size /= 2;
    }

    if (local_id == 0) {
        atomic_add(res, local_as[0]);
    }

}
