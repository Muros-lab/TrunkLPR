#pragma once

#include "types.hpp"
#include <yaml-cpp/yaml.h>

struct TrunkExtractorParams {
    // Height slice (single slice z∈[first, second]). Paper setting: [1.0, 3.0]
    std::pair<double, double> height_slice{1.0, 3.0};
    // Euclidean clustering: dbscan_eps = cluster tolerance, dbscan_min_pts = min cluster size
    double dbscan_eps;
    int dbscan_min_pts;
    // Multi-height local context density
    std::vector<std::pair<double, double>> lc_height_bands = {
        {0.5, 2.0}, {2.0, 3.5}, {3.5, 7.0}, {7.0, 10.5}
    };
    std::vector<double> lc_radii = {2.0, 3.0, 4.0, 5.0};
};

TrunkExtractorParams load_trunk_params(const YAML::Node& config);

// Extract trunks from a PCD file (single height slice z∈[1,3], diameter = cylinder_pca_rms)
std::vector<Trunk> extract_trunks(const std::string& pcd_path,
                                  const TrunkExtractorParams& params);
