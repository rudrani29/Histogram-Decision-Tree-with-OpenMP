// usage: ./hist_tree --data F --mode hist|exact --bins 256 --depth 6 --subtraction 1 --parsplit 0
//                    --threads 1 --histomp 0 --schedule static|dynamic --cutoff 2048 --out results/x.csv
#include "common.h"
#include "tree.h"
#include <omp.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

static void append_csv(const std::string &path, const std::string &header, const std::string &row)
{
    bool fresh = !std::ifstream(path).good();
    std::ofstream o(path, std::ios::app);
    if (fresh)
        o << header << "\n";
    o << row << "\n";
}

int main(int argc, char **argv)
{
    std::string data, mode = "hist", out;
    Params p;
    int bins = 256;
    for (int i = 1; i + 1 < argc; i += 2)
    {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--data")
            data = v;
        else if (k == "--mode")
            mode = v;
        else if (k == "--bins")
            bins = std::stoi(v);
        else if (k == "--depth")
            p.max_depth = std::stoi(v);
        else if (k == "--subtraction")
            p.use_subtraction = std::stoi(v);
        else if (k == "--parsplit")
            p.par_split = std::stoi(v);
        else if (k == "--minleaf")
            p.min_leaf = std::stoi(v);
        else if (k == "--threads")
            p.threads = std::stoi(v);
        else if (k == "--histomp")
            p.hist_omp = std::stoi(v);
        else if (k == "--schedule")
            p.dynamic_sched = (v == "dynamic");
        else if (k == "--cutoff")
            p.omp_cutoff = std::stoi(v);
        else if (k == "--pad")
            p.pad = std::stoi(v);
        else if (k == "--out")
            out = v;
    }
    if (data.empty())
    {
        std::cerr << "need --data\n";
        return 1;
    }
    if (bins < 2 || bins > NBINS)
    {
        std::cerr << "--bins must be in [2," << NBINS << "]\n";
        return 1;
    }
    p.max_bins = bins;
    omp_set_num_threads(p.threads);

    Dataset ds = load_raw(data);
    int n_train = (int)(0.8 * ds.N), n_test_begin = n_train;
    double bin_time = 0, train_time = 0, rtr = 0, rte = 0;
    size_t nnodes = 0;
    TrainStats st;

    if (mode == "exact")
    {
        bins = 0;
        train_time = train_exact(ds, p, n_train, n_test_begin, rtr, rte);
    }
    else
    {
        double t0 = omp_get_wtime();
        bin_dataset(ds, bins);
        bin_time = omp_get_wtime() - t0;

        // The hook: serial histogram now; build_hist_omp plugs in here later.
        HistFn hist_fn = [](const Dataset &d, const std::vector<int> &r)
        { return build_hist(d, r); };
        if (p.hist_omp)
            hist_fn = [&p](const Dataset &d, const std::vector<int> &r)
            { return build_hist_omp(d, r, p); };

        t0 = omp_get_wtime();
        Tree tree = train_tree(ds, p, hist_fn, n_train);
        train_time = omp_get_wtime() - t0;
        st = g_stats;
        nnodes = tree.nodes.size();
        rtr = rmse_range(tree, ds, 0, n_train);
        rte = rmse_range(tree, ds, n_test_begin, ds.N);
    }

    std::ostringstream row;
    row << mode << "," << bins << "," << ds.N << "," << p.threads << "," << p.max_depth << ","
        << p.use_subtraction << "," << p.par_split << "," << p.hist_omp << ","
        << (p.dynamic_sched ? "dynamic" : "static") << "," << p.omp_cutoff << "," << p.pad << ","
        << bin_time << "," << train_time << "," << st.hist_s << "," << st.split_s << "," << st.part_s << ","
        << rtr << "," << rte << "," << nnodes;
    std::cout << "mode=" << mode << " bins=" << bins << " N=" << ds.N << " threads=" << p.threads
              << " bin_time=" << bin_time << " train_time=" << train_time
              << " (hist=" << st.hist_s << " split=" << st.split_s << " part=" << st.part_s << ")"
              << " rmse_train=" << rtr << " rmse_test=" << rte << " nodes=" << nnodes << "\n";
    if (!out.empty())
        append_csv(out,
                   "mode,bins,N,threads,depth,subtraction,par_split,hist_omp,schedule,cutoff,pad,"
                   "bin_time,train_time,hist_time,split_time,part_time,rmse_train,rmse_test,nodes",
                   row.str());
}