#pragma once

#include "types.hpp"
#include <yaml-cpp/yaml.h>

struct SpectrumParams {
    // Trunk spectrum descriptor: 3D joint histogram of LC Profile × LC Concentration × sensor distance (P × R × S). Paper setting: 10 × 4 × 8 = 320D
    int bins_lc_sp_3d_profile = 10;   // P axis: LC vertical profile (band_sum/total)
    int bins_lc_sp_3d_radial  = 4;    // R axis: LC Concentration (inner/outer ratio)
    int bins_lc_sp_3d_spatial = 8;    // S axis: sensor distance (√(x²+y²))

    // Upper bounds of the 3D joint axes
    double spatial_max    = 30.0;     // S-axis upper bound
    double lc_profile_max = 1.0;      // P-axis upper bound
    double lc_radial_max  = 2.0;      // R-axis upper bound

    // Diameter-based trunk weight: raw diameter within the predefined range [min,max], decayed outside it
    double diameter_normal_min = 0.5;
    double diameter_normal_max = 1.5;

    // Sub-histograms ablation: concatenate per-axis 1D marginals instead of the 3D joint (P + R + S = 22D)
    bool descriptor_concat_1d = false;
};

// Compute the trunk spectrum descriptor (3D joint histogram, P×R×S = 320D)
void compute_spectrum(ScanData& scan, const SpectrumParams& params);

// Load Spectrum parameters from the YAML config
SpectrumParams load_spectrum_params(const YAML::Node& config);

// Chi-squared similarity: 1 / (1 + Σ(a_i - b_i)² / (a_i + b_i))
double spectrum_similarity(const ScanData& a, const ScanData& b);
