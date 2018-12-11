#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <vector>
#include <iostream>
#include <cmath>
#include <stdexcept>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)



void prefix_sum(ocl::Kernel part_prefix_sum, ocl::Kernel update_prefixes,
                gpu::gpu_mem_32u as_gpu,
                unsigned int n, unsigned int workGroupSize) {

    int size_arr = n;
    int step = 1;
    while (size_arr >= 1) {
        part_prefix_sum.exec(gpu::WorkSize(workGroupSize, (size_arr + workGroupSize - 1) / workGroupSize * workGroupSize),
                        n, step,
                        as_gpu);

        update_prefixes.exec(gpu::WorkSize(workGroupSize, (n + workGroupSize - 1) / workGroupSize * workGroupSize),
                        n, step,
                        as_gpu);

        size_arr = size_arr / workGroupSize;
        step *= workGroupSize;
    }

}

void out(gpu::gpu_mem_32u for_out, unsigned int n) {
    std::vector<unsigned int> res;
    res.resize(n);

    for_out.readN(res.data(), n);
    for (int i = 0; i < n; i++) {
        printf("%d ", res[i]);
    }
    printf("\n");
}

void radix_sort(ocl::Kernel zeros_kern,
                ocl::Kernel part_prefix_sum, ocl::Kernel update_prefixes,
                ocl::Kernel permutation, ocl::Kernel assignment,
                gpu::gpu_mem_32u as_gpu, gpu::gpu_mem_32u res_gpu,
                gpu::gpu_mem_32u zeros,
                unsigned int n, unsigned int workGroupSize) {

    for (int bit = 0; bit < 32; bit++) {
        zeros_kern.exec(gpu::WorkSize(workGroupSize, (n + workGroupSize - 1) / workGroupSize * workGroupSize),
                        as_gpu, zeros, bit, n);

        prefix_sum(part_prefix_sum, update_prefixes, zeros, n, workGroupSize);

        permutation.exec(gpu::WorkSize(workGroupSize, (n + workGroupSize - 1) / workGroupSize * workGroupSize),
                         as_gpu, res_gpu, zeros, bit, n);

        assignment.exec(gpu::WorkSize(workGroupSize, (n + workGroupSize - 1) / workGroupSize * workGroupSize),
                        res_gpu, as_gpu, n);
    }

}

int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::gpu_mem_32u as_gpu, zeros_gpu, res_gpu;
    as_gpu.resizeN(n);
    zeros_gpu.resizeN(n);
    res_gpu.resizeN(n);

    {
        ocl::Kernel part_prefix_sum(radix_kernel, radix_kernel_length, "part_prefix_sum");
        part_prefix_sum.compile();
        
        ocl::Kernel update_prefix(radix_kernel, radix_kernel_length, "update_prefix");
        update_prefix.compile();

        ocl::Kernel zeros_kern(radix_kernel, radix_kernel_length, "zeros");
        zeros_kern.compile();

        ocl::Kernel permutation(radix_kernel, radix_kernel_length, "permutation");
        permutation.compile();

        ocl::Kernel assignment(radix_kernel, radix_kernel_length, "assignment");
        assignment.compile();

        timer t;
        
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных

            unsigned int workGroupSize = 128;
            unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
            
            radix_sort(zeros_kern, part_prefix_sum, update_prefix, permutation, assignment,
                       as_gpu, res_gpu, zeros_gpu, n, workGroupSize);

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;

        res_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }
    return 0;
}
