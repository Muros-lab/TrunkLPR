#pragma once

#include <string>
#include <vector>
#include <cmath>

struct Trunk {
    double x, y;         // BEV centroid coordinates
    double diameter;     // trunk diameter
    // Multi-height Local Context: NH height bands × NR radii
    // local_context[h*NR+r] = density at height band h, radius r
    int lc_num_heights = 0;  // set at runtime
    int lc_num_radii = 0;
    std::vector<double> local_context;  // size = lc_num_heights * lc_num_radii
};

struct ScanData {
    int scan_id;
    double timestamp;
    double pose_x, pose_y;   // GT coordinates (for evaluation)
    std::string file_path;

    // Trunk extraction results
    std::vector<Trunk> trunks;

    // Spectrum descriptor
    std::vector<double> spectrum;
};

// 2D Euclidean distance
inline double euclidean_2d(double x1, double y1, double x2, double y2) {
    double dx = x1 - x2;
    double dy = y1 - y2;
    return std::sqrt(dx * dx + dy * dy);
}
