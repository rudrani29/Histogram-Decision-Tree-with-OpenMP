// usage: ./hist_tree --data F --mode hist|exact --bins 256 --depth 6 --subtraction 1 --parsplit 0 --out results/x.csv
#include "common.h"
#include <omp.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

static void append_csv(const std::string& path, const std::string& header, const std::string& row) {
    bool fresh = !std::ifstream(path).good();
    std::ofstream o(path, std::ios::app);
    if (fresh) o << header << "\n";
    o << row << "\n";
}

int main(int argc, char** argv) {
    std::string data, mode = "hist", out;
    Params p; int bins = 256;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--data") data = v; else if (k == "--mode") mode = v;
        else if (k == "--bins") bins = std::stoi(v); else if (k == "--depth") p.max_depth = std::stoi(v);
        else if (k == "--subtraction") p.use_subtraction = std::stoi(v);
        else if (k == "--parsplit") p.par_split = std::stoi(v);
        else if (k == "--minleaf") p.min_leaf = std::stoi(v);
        else if (k == "--out") out = v;
    }
    if (data.empty()) { std::cerr << "need --data\n"; return 1; }

    Dataset ds = load_raw(data);
    int n_train = (int)(0.8 * ds.N), n_test_begin = n_train;
    double bin_time = 0, train_time = 0, rtr = 0, rte = 0; size_t nnodes = 0;

    if (mode == "exact") {
        bins = 0;
        train_time = train_exact(ds, p, n_train, n_test_begin, rtr, rte);
    } else {
        double t0 = omp_get_wtime();
        bin_dataset(ds, bins);
        bin_time = omp_get_wtime() - t0;
        std::vector<int> rows(n_train);
        for (int i = 0; i < n_train; ++i) rows[i] = i;
        std::vector<TreeNode> nodes;
        t0 = omp_get_wtime();
        Hist root = build_hist(rows, ds);
        grow(ds, p, nodes, rows, root, 0);
        train_time = omp_get_wtime() - t0;
        nnodes = nodes.size();
        double se = 0;
        for (int i = 0; i < n_train; ++i) { double d = predict_bin(nodes, ds, i) - ds.y[i]; se += d * d; }
        rtr = std::sqrt(se / n_train);
        se = 0;
        for (int i = n_test_begin; i < ds.N; ++i) { double d = predict_bin(nodes, ds, i) - ds.y[i]; se += d * d; }
        rte = std::sqrt(se / (ds.N - n_test_begin));
    }
    std::ostringstream row;
    row << mode << "," << bins << "," << ds.N << "," << p.max_depth << "," << p.use_subtraction << ","
        << p.par_split << "," << bin_time << "," << train_time << "," << rtr << "," << rte << "," << nnodes;
    std::cout << "mode=" << mode << " bins=" << bins << " N=" << ds.N << " bin_time=" << bin_time
              << " train_time=" << train_time << " rmse_train=" << rtr << " rmse_test=" << rte
              << " nodes=" << nnodes << "\n";
    if (!out.empty())
        append_csv(out, "mode,bins,N,depth,subtraction,par_split,bin_time,train_time,rmse_train,rmse_test,nodes", row.str());
}
