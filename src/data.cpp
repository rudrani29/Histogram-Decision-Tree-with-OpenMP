#include "common.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

Dataset make_tiny() {
    Dataset ds; ds.N = 8; ds.D = 2;
    ds.X_bin = {0,0, 0,1, 1,0, 1,1, 2,0, 2,1, 3,0, 3,1};
    ds.y     = {10,12,11,13, 30,50,31,51};
    return ds;
}

Dataset load_raw(const std::string& path) {
    Dataset ds; FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("cannot open " + path);
    int hdr[2];
    if (fread(hdr, 4, 2, fp) != 2) throw std::runtime_error("bad header");
    ds.N = hdr[0]; ds.D = hdr[1];
    ds.X_raw.resize((size_t)ds.N * ds.D);
    ds.y.resize(ds.N);
    if (fread(ds.X_raw.data(), 4, ds.X_raw.size(), fp) != ds.X_raw.size()) throw std::runtime_error("short X");
    if (fread(ds.y.data(), 4, ds.y.size(), fp) != ds.y.size()) throw std::runtime_error("short y");
    fclose(fp); return ds;
}

void bin_dataset(Dataset& ds, int max_bins) {
    int N = ds.N, D = ds.D;
    ds.cuts.assign(D, {});
    ds.X_bin.assign((size_t)N * D, 0);

    // Phase 1: cut points, parallel over features
    #pragma omp parallel for schedule(dynamic)
    for (int f = 0; f < D; ++f) {
        std::vector<float> col(N);
        for (int i = 0; i < N; ++i) col[i] = ds.X_raw[(size_t)i * D + f];
        std::sort(col.begin(), col.end());
        std::vector<float> uniq = col;
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        auto& cuts = ds.cuts[f];
        if ((int)uniq.size() <= max_bins) {
            cuts.assign(uniq.begin() + 1, uniq.end());          // one bin per value: exact
        } else {
            for (int k = 1; k < max_bins; ++k) cuts.push_back(col[(size_t)k * N / max_bins]);
            cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
        }
    }
    // Phase 2: assign bins, parallel over rows
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
        for (int f = 0; f < D; ++f) {
            const auto& c = ds.cuts[f];
            ds.X_bin[(size_t)i * D + f] =
                (uint8_t)(std::upper_bound(c.begin(), c.end(), ds.X_raw[(size_t)i * D + f]) - c.begin());
        }
}
