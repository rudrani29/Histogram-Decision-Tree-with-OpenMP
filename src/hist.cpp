#include "common.h"

Hist build_hist(const Dataset &ds, const std::vector<int> &rows)
{
    Hist h(ds.D * 256);
    for (int i : rows)
    {
        for (int f = 0; f < ds.D; ++f)
        {
            uint8_t bin_val = ds.X_bin[(size_t)i * ds.D + f];
            Bin &b = h[f * 256 + bin_val];
            b.sum_y += ds.y[i];
            b.cnt++;
        }
    }
    return h;
}