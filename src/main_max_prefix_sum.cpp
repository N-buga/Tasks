#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include "cl/max_prefix_sum_cl.h"


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv)
{
    int benchmarkingIters = 10;
    int max_n = (1 << 24);

    for (int n = 2; n <= max_n; n *= 2) {
        std::cout << "______________________________________________" << std::endl;
        int values_range = std::min(1023, std::numeric_limits<int>::max() / n);
        std::cout << "n=" << n << " values in range: [" << (-values_range) << "; " << values_range << "]" << std::endl;

        std::vector<int> as(n, 0);
        FastRandom r(n + 1);
        for (int i = 0; i < n; ++i) {
            as[i] = (unsigned int) r.next(-values_range, values_range);
        }

        int reference_max_sum;
        int reference_result;
        {
            int max_sum = 0;
            int sum = 0;
            int result = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
                if (sum > max_sum) {
                    max_sum = sum;
                    result = i + 1;
                }
            }
            reference_max_sum = max_sum;
            reference_result = result;
        }
        std::cout << "Max prefix sum: " << reference_max_sum << " on prefix [0; " << reference_result << ")" << std::endl;

        {
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                int max_sum = 0;
                int sum = 0;
                int result = 0;
                for (int i = 0; i < n; ++i) {
                    sum += as[i];
                    if (sum > max_sum) {
                        max_sum = sum;
                        result = i + 1;
                    }
                }
                EXPECT_THE_SAME(reference_max_sum, max_sum, "CPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, result, "CPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "CPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            gpu::Device device = gpu::chooseGPUDevice(argc, argv);

            gpu::Context context;
            context.init(device.device_id_opencl);
            context.activate();

            std::vector<int> res(2, 0);

            gpu::gpu_mem_32i as_gpu, res_sum_gpu, res_pref_gpu, res_ind_gpu;
            as_gpu.resizeN(n);
            res_sum_gpu.resizeN(2*n);
            res_ind_gpu.resizeN(2*n);
            res_pref_gpu.resizeN(2*n);

            as_gpu.writeN(as.data(), n);

            ocl::Kernel prefix_sum(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "max_prefix_sum");
            prefix_sum.compile();

            timer t;

            unsigned int workGroupSize = 128;

            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                prefix_sum.exec(gpu::WorkSize(workGroupSize, (n + workGroupSize - 1) / workGroupSize * workGroupSize),
                                n, 0,
                                as_gpu, as_gpu, res_ind_gpu, true,
                                res_sum_gpu, res_pref_gpu, res_ind_gpu);
                int size_arr = n/workGroupSize;
                int offset = 0;
                while (size_arr > 1) {
                    prefix_sum.exec(gpu::WorkSize(workGroupSize, (size_arr + workGroupSize - 1) / workGroupSize * workGroupSize),
                                    size_arr, offset,
                                    res_sum_gpu, res_pref_gpu, res_ind_gpu, false,
                                    res_sum_gpu, res_pref_gpu, res_ind_gpu);

                    offset += size_arr;
                    size_arr = int(ceil(1.0*size_arr/workGroupSize));
                }

                int ans_ind;
                int ans_pref;

                res_ind_gpu.readN(&ans_ind, 1, offset);
                res_pref_gpu.readN(&ans_pref, 1, offset);

                if (ans_pref < 0) {
                    ans_pref = 0;
                    ans_ind = 0;
                } else {
                    ans_ind += 1;
                }

                EXPECT_THE_SAME(reference_max_sum, ans_pref, "GPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, ans_ind, "GPU result should be consistent!");

                t.nextLap();
            }

            std::cout << "GPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
