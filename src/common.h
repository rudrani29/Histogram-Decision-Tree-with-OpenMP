#pragma once
#include <vector>
#include <string>
#include <cstdint>

constexpr int NBINS = 256; // bins per feature (uint8_t index)

struct Bin
{
    double sum_y = 0;
    uint32_t cnt = 0;
};
using Hist = std::vector<Bin>; // size D*NBINS, index f*NBINS+b

struct Dataset
{
    int N = 0, D = 0;
    std::vector<float> X_raw;   // N*D floats (exact tree needs these)
    std::vector<uint8_t> X_bin; // N*D bins, X_bin[i*D+f]
    std::vector<float> y;
    std::vector<std::vector<float>> cuts; // cut points per feature
};

struct Split
{
    double gain = -1;
    int feature = -1, bin = -1;
};

struct Params
{
    // shared / Person A
    int max_depth = 6, min_leaf = 20;
    double min_gain = 1e-7;
    bool use_subtraction = true, par_split = false;
    int max_bins = 256;
    // Person B (histogram parallelism)
    bool hist_omp = false;      // use build_hist_omp instead of build_hist
    int threads = 1;            // OpenMP threads
    bool dynamic_sched = false; // static (default) vs dynamic row scheduling
    bool pad = true;            // 64-byte-aligned private histograms (false = deliberately misaligned)
    int omp_cutoff = 2048;      // nodes with fewer rows than this use the serial build_hist
};

// ---- Person A ----
Split best_for_feature(const Hist &h, int f, int min_leaf);
Split find_best_split(const Hist &h, int D, int min_leaf);
Split find_best_split_omp(const Hist &h, int D, int min_leaf);
Hist subtract(const Hist &parent, const Hist &small);
Dataset make_tiny();
Dataset load_raw(const std::string &path);
void bin_dataset(Dataset &ds, int max_bins);
double train_exact(const Dataset &ds, const Params &p, int n_train, int n_test_begin,
                   double &rmse_train, double &rmse_test);

// ---- Person B ----  (argument order: dataset first, then rows)
Hist build_hist(const Dataset &ds, const std::vector<int> &rows);
Hist build_hist_omp(const Dataset &ds, const std::vector<int> &rows, const Params &p);