#include "common.h"
#include <omp.h>

int main()
{
    std::cout << "OpenMP Max Threads: " << omp_get_max_threads() << std::endl;
    std::cout << "---------------------------------------" << std::endl;

    // 8-row toy dataset (1 feature, 8 rows, 4 bins)
    Dataset ds;
    ds.N = 8;
    ds.D = 1;
    ds.X_bin = {0, 0, 1, 1, 2, 2, 3, 3}; // 8 values for 1 feature
    ds.y = {10.0f, 12.0f, 11.0f, 13.0f, 30.0f, 32.0f, 31.0f, 33.0f};

    std::vector<int> rows = {0, 1, 2, 3, 4, 5, 6, 7};

    // Run build_hist
    Hist h = build_hist(ds, rows);

    // Verify output for feature 0 (bins 0 to 3)
    bool passed = true;
    for (int b = 0; b < 4; ++b)
    {
        Bin bin_data = h[0 * 256 + b];
        std::cout << "Bin " << b << ": Count = " << bin_data.cnt
                  << ", Sum_Y = " << bin_data.sum_y << std::endl;
    }

    return 0;
}