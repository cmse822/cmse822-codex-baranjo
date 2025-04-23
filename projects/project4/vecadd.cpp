#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <cmath>
#include <cstdlib>

constexpr int N = 100000;
constexpr double TOL = 1e-4;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <number of threads>\n";
        return 1;
    }

    int N = std::atoi(argv[1]);
    if (N <= 0) {
        std::cerr << "Error: N must be a positive integer.\n";
        return 1;
    }

    int num_devices = omp_get_num_devices();
    std::cout << "OpenMP target devices available: " << num_devices << "\n";

    std::vector<double> a(N), b(N), c(N), res(N);

    // Fill the arrays using simple for loops.
    for (int i = 0; i < N; ++i) {
        a[i] = static_cast<double>(i);
        b[i] = 2.0f * static_cast<double>(i);
        res[i] = a[i] + b[i];
        c[i] = 0.0;
    }

    double* a_ptr = a.data();
    double* b_ptr = b.data();
    double* c_ptr = c.data();

    // Add two vectors using a simple loop and measure time.
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp target loop \
        map(to: a_ptr[0:N], b_ptr[0:N]) \
        map(from: c_ptr[0:N])
    for (int i = 0; i < N; ++i) {
        c_ptr[i] = a_ptr[i] + b_ptr[i];
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Test results.
    int err = 0;    
    for (int i = 0; i < N; ++i) {
        double diff = c[i] - res[i];
        if (diff * diff > TOL) {
            ++err;
        }
    }

    std::cout << "Vectors added with " << err << " errors\n";
    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
    double flops = 2.0 * N / elapsed.count();
    std::cout << "FLOP rate: " << flops << " FLOP/s\n";
    return 0;
}
