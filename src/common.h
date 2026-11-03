#pragma once
#include <vector>
#include <cstdint>
#include <iostream>

struct Bin
{
    double sum_y = 0.0;
    uint32_t cnt = 0;
};

using Hist = std::vector<Bin>; // Size: D * 256, index: f * 256 + b

struct Dataset
{
    int N, D;
    std::vector<uint8_t> X_bin;
    std::vector<float> y;
};

// Computes standard 1D/2D serial histogram
Hist build_hist(const Dataset &ds, const std::vector<int> &rows);