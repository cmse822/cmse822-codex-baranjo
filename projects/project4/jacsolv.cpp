#include <omp.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "mm_utils.hpp"

constexpr double TOLERANCE = 0.001;
constexpr int DEF_SIZE = 1000;
constexpr int MAX_ITERS = 100000;
constexpr double LARGE = 1000000.0;

#include <iostream>
int main(int argc, char** argv) {
    int Ndim;
    int iters;

    double start_time, elapsed_time;
    // TYPE err, chksum;

    if (argc == 2) {
        Ndim = std::atoi(argv[1]);
    } else {
        Ndim = DEF_SIZE;
    }

    std::cout << "ndim = " << Ndim << "\n";

    std::vector<TYPE> A(Ndim * Ndim);
    std::vector<TYPE> b(Ndim);
    std::vector<TYPE> xnew(Ndim);
    std::vector<TYPE> xold(Ndim);

    initDiagDomNearIdentityMatrix(Ndim, A.data());

#ifdef VERBOSE
#endif

    for (int i = 0; i < Ndim; ++i) {
        xold[i] = static_cast<TYPE>(0.0);
        xnew[i] = static_cast<TYPE>(0.0);
        b[i] = static_cast<TYPE>(std::rand() % 51) / 100.0;
    }

    auto *xnew_dat = xnew.data();
    auto *xold_dat = xold.data();
    auto *A_dat = A.data();
    auto *b_dat = b.data();

    start_time = omp_get_wtime();

    std::cout << "Max number of threads: " << omp_get_max_threads() << "\n";

    TYPE conv = LARGE;
    #pragma omp target data \
        map(to: A_dat[0:Ndim*Ndim], b_dat[0:Ndim]) \
        map(tofrom: xold_dat[0:Ndim], xnew_dat[0:Ndim])
    {
        iters = 0;
        while (conv > TOLERANCE && iters < MAX_ITERS) {
            ++iters;

            #pragma omp target loop
            for (int i = 0; i < Ndim; ++i) {
                TYPE sum = TYPE{0};
                for (int j = 0; j < Ndim; ++j) {
                    if (i != j) sum += A_dat[i * Ndim + j] * xold_dat[j];
                }
                xnew_dat[i] = (b_dat[i] - sum) / A_dat[i * Ndim + i];
            }

            conv = TYPE{0};
            #pragma omp target loop map(tofrom : conv)
            for (int i = 0; i < Ndim; ++i) {
                TYPE d = xnew_dat[i] - xold_dat[i];
                conv += d * d;
            }
            conv = std::sqrt(conv);

            #pragma omp target loop
            for (int i = 0; i < Ndim; ++i) {
                TYPE tmp = xold_dat[i];
                xold_dat[i] = xnew_dat[i];
                xnew_dat[i] = tmp;
            }
        }
    }

    elapsed_time = omp_get_wtime() - start_time;
    std::cout << "Convergence = " << static_cast<float>(conv) << " with " << iters << " iterations and " << elapsed_time << " seconds\n";

    double totalFlops = 2.0 * static_cast<double>(Ndim) * static_cast<double>(Ndim) * static_cast<double>(iters);

    double flopsPerSecond = (elapsed_time > 0) ? totalFlops / elapsed_time : 0.0;

    std::cout << "FLOP rate: " << flopsPerSecond << " FLOP/s\n";
    return 0;
}