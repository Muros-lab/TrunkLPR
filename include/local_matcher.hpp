#pragma once

#include "types.hpp"
#include <yaml-cpp/yaml.h>

// RANSAC parameters
struct BFRansacParams {
    int ransac_iters = 1000;
    double inlier_thresh = 1.5;        // RANSAC inlier threshold (m, = τ). Also used as the distance-compatibility tolerance for hypothesis pairs (pair_dist_tol)
    double density_compat_ratio = 0.50; // hard density-compatibility filter: passes if min/max >= 1 - value (0 disables the filter)
    double density_sigma_d = 0.10;     // soft Gaussian density weight: exp(-(1-ds)²/(2σ_d²))
    bool use_spread = true;            // enable Spatial Spread weighting
};

// Load RANSAC parameters from the YAML config
BFRansacParams load_ransac_params(const YAML::Node& config);

// RANSAC scoring
double ransac_score(const ScanData& query, const ScanData& db,
                    const BFRansacParams& params);
