// TEMPORARY find_best_split (section 9 of the doc).
// DELETE this file when Person A's real split.cpp arrives (else duplicate symbol).
#include "common.h"
#include <algorithm>
Split find_best_split(const Hist &h, int D, int min_leaf)
{
    Split best;
    min_leaf = std::max(min_leaf, 1);
    for (int f = 0; f < D; ++f)
    {
        const Bin *hf = &h[f * 256];
        double S = 0;
        uint32_t n = 0;
        for (int b = 0; b < 256; ++b)
        {
            S += hf[b].sum_y;
            n += hf[b].cnt;
        }
        double S_L = 0;
        uint32_t n_L = 0;
        for (int b = 0; b < 255; ++b)
        {
            S_L += hf[b].sum_y;
            n_L += hf[b].cnt;
            uint32_t n_R = n - n_L;
            if ((int)n_L < min_leaf || (int)n_R < min_leaf)
                continue;
            double S_R = S - S_L;
            double gain = S_L * S_L / n_L + S_R * S_R / n_R - S * S / n;
            if (gain > best.gain)
                best = {gain, f, b};
        }
    }
    return best;
}