#pragma once

#include "types.hpp"
#include <yaml-cpp/yaml.h>
#include <vector>
#include <functional>

struct EvalParams {
    double world_thresh;
    double time_thresh;
    int spectrum_top_n = 50;                 // number of retrieved candidates (K in the paper)
};

struct EvalResult {
    double recall_1;
    double f1_max;
    int num_revisits;
    int num_correct;
    int num_queries;
};

EvalParams load_eval_params(const YAML::Node& config);

// Callback type: (query_scan, db_scan) -> score [0, 1] (shared by global similarity and local RANSAC)
using SimilarityFunc = std::function<double(const ScanData&, const ScanData&)>;

// Spectrum top-K → RANSAC Reranking
EvalResult evaluate_rerank(const std::vector<ScanData>& scans,
                            const EvalParams& eval_params,
                            SimilarityFunc global_sim_fn,
                            SimilarityFunc ransac_fn);
