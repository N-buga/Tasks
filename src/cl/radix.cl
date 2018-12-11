#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 128


void prefix_sum_local(__local unsigned int* prefix_sum, int arr_size, int local_id) {

    for (int i = 1; i < arr_size; i *= 2) {
        int cur_index = (((local_id)/i) + 1)*i + local_id;
        int add_index = ((cur_index) / i)*i - 1;

        if (cur_index < arr_size && add_index >= 0) {
            prefix_sum[cur_index] += prefix_sum[add_index];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

void radix_local(__local unsigned int* as, __local int* place, __local unsigned int* res,
                 int* total_falses, int* total_true,
                 int arr_size, int local_id, int bit) {
    place[local_id] = as[local_id] ^ (1 << bit);

    barrier(CLK_LOCAL_MEM_FENCE);

    prefix_sum_local(place, arr_size, local_id);
    total_falses[0] = place[arr_size - 1];
    total_true[0] = arr_size - total_falses[0];

    if ((as[local_id] ^ (1 << bit)) == 0) {
        place[local_id] = local_id - place[local_id] + total_falses[0];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    res[place[local_id]] = as[local_id];
}

__kernel void part_prefix_sum(
                           unsigned int n,
                           unsigned int step,
                  __global unsigned int* arr)
{
    const unsigned int global_id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);

    unsigned int offset = step - 1;

    __local unsigned int local_res_prefix[WORK_GROUP_SIZE];

    if (global_id*step + offset < n) {
        local_res_prefix[local_id] = arr[global_id*step + offset];
    } else {
        local_res_prefix[local_id] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    prefix_sum_local(local_res_prefix, WORK_GROUP_SIZE, local_id);

    barrier(CLK_LOCAL_MEM_FENCE);

    if (global_id * step + offset < n) {
        arr[global_id * step + offset] = local_res_prefix[local_id];
    }
}

__kernel void update_prefix(
        unsigned int n,
        unsigned int step,
        __global unsigned int * arr
) {
    const unsigned int global_id = get_global_id(0);
    int add_index = (global_id / step) * step - 1;

    unsigned int offset = step - 1;

    if ((global_id - offset) % step != 0 && global_id < n && add_index >= 0 && global_id % (step * WORK_GROUP_SIZE) >= step) {
        arr[global_id] += arr[add_index];
    }
}

__kernel void zeros(
        __global const unsigned int* arr,
        __global unsigned int* zeros,
        unsigned int bit,
        unsigned int n
) {
    const unsigned int global_id = get_global_id(0);

    if (global_id < n) {
        zeros[global_id] = 1 - (unsigned int) ((arr[global_id] & (1 << bit)) != 0);
    }
}

__kernel void permutation(
        __global const unsigned int* arr,
        __global       unsigned int* res,
        __global const unsigned int* zeros,
        unsigned int bit,
        unsigned int n
) {
    const unsigned int global_id = get_global_id(0);

    if (global_id < n) {
        if ((arr[global_id] & (1 << bit)) != 0) {
            int cur_ones = global_id - zeros[global_id] + 1;

            unsigned int index = zeros[n - 1] + cur_ones - 1;
            res[index] = arr[global_id];
        } else {
            unsigned int index = zeros[global_id] - 1;
            res[index] = arr[global_id];
        }
    }
}

__kernel void assignment(
        __global const unsigned int* arr_from,
        __global       unsigned int* arr_to,
        unsigned int n
) {
    const unsigned int global_id = get_global_id(0);

    if (global_id < n) {
        arr_to[global_id] = arr_from[global_id];
    }
}
