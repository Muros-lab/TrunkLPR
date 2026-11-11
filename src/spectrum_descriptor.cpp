#include "spectrum_descriptor.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

SpectrumParams load_spectrum_params(const YAML::Node& config) {
    SpectrumParams p;
    auto sp = config["spectrum"];
    p.spatial_max = sp["spatial_max"].as<double>();
    p.lc_profile_max = sp["lc_profile_max"] ? sp["lc_profile_max"].as<double>() : 1.0;
    p.lc_radial_max  = sp["lc_radial_max"]  ? sp["lc_radial_max"].as<double>()  : 2.0;
    p.bins_lc_sp_3d_profile = sp["bins_lc_sp_3d_profile"] ? sp["bins_lc_sp_3d_profile"].as<int>() : 10;
    p.bins_lc_sp_3d_radial  = sp["bins_lc_sp_3d_radial"]  ? sp["bins_lc_sp_3d_radial"].as<int>()  : 4;
    p.bins_lc_sp_3d_spatial = sp["bins_lc_sp_3d_spatial"] ? sp["bins_lc_sp_3d_spatial"].as<int>() : 8;
    p.diameter_normal_min = sp["diameter_normal_min"] ? sp["diameter_normal_min"].as<double>() : 0.5;
    p.diameter_normal_max = sp["diameter_normal_max"] ? sp["diameter_normal_max"].as<double>() : 1.5;
    p.descriptor_concat_1d = sp["descriptor_concat_1d"] ? sp["descriptor_concat_1d"].as<bool>() : false;
    return p;
}

// Weighted 3D joint histogram: (LC Profile, LC Concentration, sensor distance) 3-tuples are
// accumulated by weight on a bx×by×bz grid, then L1-normalized and flattened (ix*(by*bz)+iy*bz+iz)
static std::vector<double> build_weighted_3d_histogram(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    const std::vector<double>& zs,
    const std::vector<double>& weights,
    int bx, int by, int bz,
    double xmin, double xmax,
    double ymin, double ymax,
    double zmin, double zmax)
{
    if (bx <= 0 || by <= 0 || bz <= 0 || xs.empty()) return std::vector<double>();
    if (xs.size() != ys.size() || xs.size() != zs.size()) return std::vector<double>();
    if (xmax <= xmin || ymax <= ymin || zmax <= zmin) return std::vector<double>();

    double rx = xmax - xmin, ry = ymax - ymin, rz = zmax - zmin;
    int total = bx * by * bz;
    std::vector<double> hist(total, 0.0);
    for (size_t i = 0; i < xs.size(); i++) {
        int ix = static_cast<int>((xs[i] - xmin) / rx * bx);
        int iy = static_cast<int>((ys[i] - ymin) / ry * by);
        int iz = static_cast<int>((zs[i] - zmin) / rz * bz);
        if (ix < 0) ix = 0; if (ix >= bx) ix = bx - 1;
        if (iy < 0) iy = 0; if (iy >= by) iy = by - 1;
        if (iz < 0) iz = 0; if (iz >= bz) iz = bz - 1;
        double w = (i < weights.size()) ? weights[i] : 1.0;
        hist[ix * (by * bz) + iy * bz + iz] += w;
    }
    double sum = std::accumulate(hist.begin(), hist.end(), 0.0);
    if (sum > 0) {
        for (auto& v : hist) v /= sum;
    }
    return hist;
}

void compute_spectrum(ScanData& scan, const SpectrumParams& params) {
    int N = static_cast<int>(scan.trunks.size());
    scan.spectrum.clear();
    if (N == 0) return;

    // Local context dimensions (taken from the first non-empty trunk)
    int lc_nh = 0, lc_nr = 0;
    for (int i = 0; i < N; i++) {
        if (!scan.trunks[i].local_context.empty()) {
            lc_nh = scan.trunks[i].lc_num_heights;
            lc_nr = scan.trunks[i].lc_num_radii;
            break;
        }
    }

    // Diameter-based trunk weight
    // Within the predefined diameter range [dmin,dmax]: w = d (raw diameter); outside it: piecewise linear decay with scale = dmax-dmin.
    // Lower bound 0 (prevents negative weights; same as Eq. (8) in the paper).
    std::vector<double> all_weights(N, 1.0);
    {
        const double dmin = params.diameter_normal_min;
        const double dmax = params.diameter_normal_max;
        const double scale = dmax - dmin;
        for (int i = 0; i < N; i++) {
            double d = scan.trunks[i].diameter;
            double w;
            if (d < dmin && scale > 1e-9) {
                double rel = (dmin - d) / scale;
                w = dmin * std::max(0.0, 1.0 - rel);
            } else if (d > dmax && scale > 1e-9) {
                double rel = (d - dmax) / scale;
                w = dmax * std::max(0.0, 1.0 - rel);
            } else {
                w = d;
            }
            all_weights[i] = std::max(0.0, w);
        }
    }

    // 3D joint histogram (LC Profile × LC Concentration × sensor distance)
    // Per (trunk, band): accumulate the 3-tuple (LC Profile=band_sum/total, LC Concentration=inner/outer, sensor distance=√(x²+y²))
    std::vector<double> lcsp_pvals, lcsp_rvals, lcsp_svals, lcsp_wts;
    if (lc_nh > 0 && lc_nr >= 2) {
        lcsp_pvals.reserve(N * lc_nh);
        lcsp_rvals.reserve(N * lc_nh);
        lcsp_svals.reserve(N * lc_nh);
        lcsp_wts.reserve(N * lc_nh);
        for (int i = 0; i < N; i++) {
            const auto& lc = scan.trunks[i].local_context;
            if (lc.empty()) continue;
            double total = 0.0;
            for (double v : lc) total += v;
            if (total < 1e-12) continue;
            double dw = all_weights[i];
            double xi = scan.trunks[i].x, yi = scan.trunks[i].y;
            double sp_val = std::sqrt(xi * xi + yi * yi);
            for (int h = 0; h < lc_nh; h++) {
                double band_sum = 0.0;
                for (int r = 0; r < lc_nr; r++) {
                    band_sum += lc[h * lc_nr + r];
                }
                double inner = lc[h * lc_nr + 0];
                double outer = lc[h * lc_nr + (lc_nr - 1)];
                if (outer > 1e-12) {
                    lcsp_pvals.push_back(band_sum / total);
                    lcsp_rvals.push_back(inner / outer);
                    lcsp_svals.push_back(sp_val);
                    lcsp_wts.push_back(dw);
                }
            }
        }
    }
    auto hist_lc_sp_3d_joint = build_weighted_3d_histogram(
        lcsp_pvals, lcsp_rvals, lcsp_svals, lcsp_wts,
        params.bins_lc_sp_3d_profile, params.bins_lc_sp_3d_radial, params.bins_lc_sp_3d_spatial,
        0, params.lc_profile_max,
        0, params.lc_radial_max,
        0, params.spatial_max);

    if (!params.descriptor_concat_1d) {
        scan.spectrum.insert(scan.spectrum.end(),
                             hist_lc_sp_3d_joint.begin(), hist_lc_sp_3d_joint.end());
        return;
    }

    // Sub-histograms ablation: sum the per-axis marginals of the 3D joint (same samples and weights)
    // and concatenate them (P + R + S = 22D). Because the joint is L1-normalized, each marginal sums to 1.
    if (hist_lc_sp_3d_joint.empty()) return;
    const int bp = params.bins_lc_sp_3d_profile;
    const int br = params.bins_lc_sp_3d_radial;
    const int bs = params.bins_lc_sp_3d_spatial;
    std::vector<double> mp(bp, 0.0), mr(br, 0.0), ms(bs, 0.0);
    for (int ix = 0; ix < bp; ix++) {
        for (int iy = 0; iy < br; iy++) {
            for (int iz = 0; iz < bs; iz++) {
                double v = hist_lc_sp_3d_joint[ix * (br * bs) + iy * bs + iz];
                mp[ix] += v;
                mr[iy] += v;
                ms[iz] += v;
            }
        }
    }
    scan.spectrum.insert(scan.spectrum.end(), mp.begin(), mp.end());
    scan.spectrum.insert(scan.spectrum.end(), mr.begin(), mr.end());
    scan.spectrum.insert(scan.spectrum.end(), ms.begin(), ms.end());
}

double spectrum_similarity(const ScanData& a, const ScanData& b) {
    if (a.spectrum.empty() || b.spectrum.empty()) return 0.0;
    if (a.spectrum.size() != b.spectrum.size()) return 0.0;

    const int D = static_cast<int>(a.spectrum.size());
    const auto& sa = a.spectrum;
    const auto& sb = b.spectrum;

    // Chi-squared distance to similarity: 1 / (1 + chi2)
    double chi2 = 0.0;
    for (int i = 0; i < D; i++) {
        double sum_ab = sa[i] + sb[i];
        if (sum_ab > 1e-12) {
            double diff = sa[i] - sb[i];
            chi2 += diff * diff / sum_ab;
        }
    }
    return 1.0 / (1.0 + chi2);
}
