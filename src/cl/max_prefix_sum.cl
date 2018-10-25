#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 128

void max_prefix_local(__local int* sum, __local int* prefix_sum, __local int* ind, int arr_size,
                      int local_id, int global_id) {
    int res_ind = -1;
    int res_max = -1;

    for (int i = 1; i < arr_size; i *= 2) {
        if (2*i*local_id < arr_size) {
            if (sum[2*i*local_id] + prefix_sum[2*i*local_id + i] > prefix_sum[2*i*local_id]) {
                ind[2*i*local_id] = ind[2*i*local_id + i];
                prefix_sum[2*i*local_id] = sum[2*i*local_id] + prefix_sum[2*i*local_id + i];
            }
            sum[2*i*local_id] += sum[2*i*local_id + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void max_prefix_sum(
                           unsigned int n,
                           unsigned int offset,
                  __global const int* sum,
                  __global const int* pref,
                  __global const int* ind,
                           bool first_iteration,
                  __global int* res_sum,
                  __global int* res_pref,
                  __global int* res_ind)
{
    const unsigned int global_id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);

    __local int local_res_sum[WORK_GROUP_SIZE];
    __local int local_res_prefix[WORK_GROUP_SIZE];
    __local int local_ind[WORK_GROUP_SIZE];

    if (global_id < n) {
        local_res_sum[local_id] = sum[global_id + offset];
        local_res_prefix[local_id] = pref[global_id + offset];
    } else {
        local_res_sum[local_id] = 0;
        local_res_prefix[local_id] = 0;
    }

    if (first_iteration) {
        local_ind[local_id] = global_id;
    } else {
        local_ind[local_id] = ind[global_id + offset];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    max_prefix_local(local_res_sum, local_res_prefix, local_ind, WORK_GROUP_SIZE, local_id, global_id);

    if (local_id == 0) {
        int new_offset = 0;
        if (!first_iteration) {
            new_offset = n;
        }
        res_sum[offset + new_offset + global_id/WORK_GROUP_SIZE] = local_res_sum[0];
        res_pref[offset + new_offset + global_id/WORK_GROUP_SIZE] = local_res_prefix[0];
        res_ind[offset + new_offset + global_id/WORK_GROUP_SIZE] = local_ind[0];
    }
}
