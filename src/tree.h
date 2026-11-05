#pragma once
#include <functional>
#include <vector>
#include "common.h"

struct Node
{
    int feature = -1, bin = -1; // split: go left if X_bin[row][feature] <= bin
    int left = -1, right = -1;  // child indices into Tree::nodes, -1 for a leaf
    float value = 0;            // mean y (the prediction if this is a leaf)
    int depth = 0;
    bool is_leaf() const { return left == -1; }
};

struct Tree
{
    std::vector<Node> nodes;
}; // nodes[0] is the root

// The "hook": which function builds a histogram. Swap in build_hist_omp later.
using HistFn = std::function<Hist(const Dataset &, const std::vector<int> &)>;

// Seconds spent in each phase during the last train_tree call (for the Amdahl discussion).
struct TrainStats
{
    double hist_s = 0, split_s = 0, part_s = 0;
};
extern TrainStats g_stats;

void partition(const Dataset &ds, const std::vector<int> &rows, int feature, int bin,
               std::vector<int> &left, std::vector<int> &right);

// Trains on rows [0, n_train). n_train < 0 means all N rows.
Tree train_tree(const Dataset &ds, const Params &p, HistFn hist_fn, int n_train = -1);

float predict_row(const Tree &t, const Dataset &ds, int row);
std::vector<float> predict_all(const Tree &t, const Dataset &ds);
double rmse(const std::vector<float> &pred, const std::vector<float> &y);
// RMSE of the tree over rows [begin, end)
double rmse_range(const Tree &t, const Dataset &ds, int begin, int end);
void print_tree(const Tree &t, int node = 0, int indent = 0);