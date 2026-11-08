#include "local_matcher.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>


BFRansacParams load_ransac_params(const YAML::Node& config) {
    auto rs = config["ransac"];
    BFRansacParams p;
    p.ransac_iters = rs["ransac_iters"].as<int>();
    p.inlier_thresh = rs["inlier_thresh"].as<double>();
    p.density_compat_ratio = rs["density_compat_ratio"].as<double>();
    if (rs["density_sigma_d"])
        p.density_sigma_d = rs["density_sigma_d"].as<double>();
    if (rs["use_spread"])
        p.use_spread = rs["use_spread"].as<bool>();
    return p;
}

// Estimate a 2D rigid transform (rotation + translation) from two matched pairs
static bool estimate_rigid_2d(
    double qx0, double qy0, double dx0, double dy0,
    double qx1, double qy1, double dx1, double dy1,
    double& cos_theta, double& sin_theta, double& tx, double& ty) {

    double dqx = qx1 - qx0;
    double dqy = qy1 - qy0;
    double ddx = dx1 - dx0;
    double ddy = dy1 - dy0;

    double q_len = std::sqrt(dqx * dqx + dqy * dqy);
    double d_len = std::sqrt(ddx * ddx + ddy * ddy);

    if (q_len < 1e-6 || d_len < 1e-6) return false;

    double angle_q = std::atan2(dqy, dqx);
    double angle_d = std::atan2(ddy, ddx);
    double theta = angle_d - angle_q;

    cos_theta = std::cos(theta);
    sin_theta = std::sin(theta);

    tx = dx0 - (cos_theta * qx0 - sin_theta * qy0);
    ty = dy0 - (sin_theta * qx0 + cos_theta * qy0);

    return true;
}


// Multi-scale density similarity: mean min/max ratio of local_context
static double density_sim(const Trunk& a, const Trunk& b) {
    int n = static_cast<int>(std::min(a.local_context.size(), b.local_context.size()));
    std::vector<double> ratios;
    ratios.reserve(n);
    for (int s = 0; s < n; s++) {
        double max_d = std::max(a.local_context[s], b.local_context[s]);
        if (max_d > 1e-6) {
            ratios.push_back(std::min(a.local_context[s], b.local_context[s]) / max_d);
        }
    }
    // When neither trunk has any local_context evidence (about 0 at every scale), treat them as fully similar (1.0) to avoid false rejection
    if (ratios.empty()) return 1.0;
    double mean = 0;
    for (double r : ratios) mean += r;
    mean /= ratios.size();
    return mean;
}

// Compute the forward score: hard inlier threshold + density/spread weighting
static double compute_forward_score(
    const ScanData& query, const ScanData& db,
    double cos_t, double sin_t, double tx, double ty,
    double thresh_sq, const BFRansacParams& params) {

    int Nq = static_cast<int>(query.trunks.size());
    int Nd = static_cast<int>(db.trunks.size());

    double two_sigma_d_sq = (params.density_sigma_d > 0)
        ? 2.0 * params.density_sigma_d * params.density_sigma_d : 0.0;

    // Spatial Spread: track inlier positions
    double sp_sx = 0, sp_sy = 0, sp_sxx = 0, sp_syy = 0;

    double weighted_sum = 0;
    int n_inliers = 0;
    for (int i = 0; i < Nq; i++) {
        double rx = cos_t * query.trunks[i].x - sin_t * query.trunks[i].y + tx;
        double ry = sin_t * query.trunks[i].x + cos_t * query.trunks[i].y + ty;
        double best_dist_sq = 1e18;
        int best_j = -1;
        for (int j = 0; j < Nd; j++) {
            double ex = rx - db.trunks[j].x;
            double ey = ry - db.trunks[j].y;
            double d2 = ex * ex + ey * ey;
            if (d2 < best_dist_sq) {
                best_dist_sq = d2;
                best_j = j;
            }
        }
        if (best_j < 0 || best_dist_sq > thresh_sq) continue;

        // Density Gaussian weighting
        double d_weight = 1.0;
        if (two_sigma_d_sq > 0) {
            double sim = density_sim(query.trunks[i], db.trunks[best_j]);
            double err = 1.0 - sim;
            d_weight = std::exp(-err * err / two_sigma_d_sq);
        }

        weighted_sum += d_weight;
        n_inliers++;

        // Spatial Spread: accumulate inlier positions
        if (params.use_spread) {
            sp_sx += query.trunks[i].x;
            sp_sy += query.trunks[i].y;
            sp_sxx += query.trunks[i].x * query.trunks[i].x;
            sp_syy += query.trunks[i].y * query.trunks[i].y;
        }
    }

    double base_score = weighted_sum / std::max(Nq, Nd);

    // Spatial Spread: inlier variance / variance of all trunks
    if (params.use_spread && n_inliers >= 3) {
        double mx = sp_sx / n_inliers, my = sp_sy / n_inliers;
        double inlier_var = (sp_sxx / n_inliers - mx * mx) + (sp_syy / n_inliers - my * my);
        double total_sx = 0, total_sy = 0, total_sxx = 0, total_syy = 0;
        for (int i = 0; i < Nq; i++) {
            total_sx += query.trunks[i].x; total_sy += query.trunks[i].y;
            total_sxx += query.trunks[i].x * query.trunks[i].x;
            total_syy += query.trunks[i].y * query.trunks[i].y;
        }
        double tmx = total_sx / Nq, tmy = total_sy / Nq;
        double total_var = (total_sxx / Nq - tmx * tmx) + (total_syy / Nq - tmy * tmy);
        if (total_var > 1e-6) {
            double spread_ratio = std::min(inlier_var / total_var, 1.0);
            base_score *= spread_ratio;
        }
    }

    return base_score;
}

// RANSAC scoring
double ransac_score(const ScanData& query, const ScanData& db,
                    const BFRansacParams& params) {
    int Nq = static_cast<int>(query.trunks.size());
    int Nd = static_cast<int>(db.trunks.size());

    if (Nq < 3 || Nd < 3) return 0.0;

    // Precompute the inlier threshold. The distance-compatibility tolerance for hypothesis pairs equals inlier_thresh (= τ).
    double thresh_sq = params.inlier_thresh * params.inlier_thresh;
    double pair_dist_tol = params.inlier_thresh;
    // Precompute the pairs that satisfy density compatibility
    std::vector<std::vector<int>> compat(Nq);
    for (int i = 0; i < Nq; i++) {
        for (int j = 0; j < Nd; j++) {
            if (params.density_compat_ratio > 0) {
                bool density_ok = true;
                double thresh = 1.0 - params.density_compat_ratio;
                const auto& qlc = query.trunks[i].local_context;
                const auto& dlc = db.trunks[j].local_context;
                int n = std::min(qlc.size(), dlc.size());
                for (int s = 0; s < n; s++) {
                    double max_den = std::max(qlc[s], dlc[s]);
                    if (max_den > 1e-6) {
                        double ratio = std::min(qlc[s], dlc[s]) / max_den;
                        if (ratio < thresh) { density_ok = false; break; }
                    }
                }
                if (!density_ok) continue;
            }

            compat[i].push_back(j);
        }
    }

    std::mt19937 rng(static_cast<uint32_t>(
        static_cast<uint64_t>(query.scan_id) * 10007 + static_cast<uint64_t>(db.scan_id)));
    std::uniform_int_distribution<int> dist_q(0, Nq - 1);

    double best_score = 0.0;

    for (int iter = 0; iter < params.ransac_iters; iter++) {
        int qi0 = dist_q(rng);
        if (compat[qi0].empty()) continue;

        std::uniform_int_distribution<int> d0_dist(0, static_cast<int>(compat[qi0].size()) - 1);
        int di0 = compat[qi0][d0_dist(rng)];

        int qi1 = dist_q(rng);
        if (qi1 == qi0) continue;

        double q_dist = std::sqrt(
            (query.trunks[qi1].x - query.trunks[qi0].x) * (query.trunks[qi1].x - query.trunks[qi0].x) +
            (query.trunks[qi1].y - query.trunks[qi0].y) * (query.trunks[qi1].y - query.trunks[qi0].y));

        std::vector<int> di1_cands;
        for (int dj : compat[qi1]) {
            if (dj == di0) continue;
            double d_dist = std::sqrt(
                (db.trunks[dj].x - db.trunks[di0].x) * (db.trunks[dj].x - db.trunks[di0].x) +
                (db.trunks[dj].y - db.trunks[di0].y) * (db.trunks[dj].y - db.trunks[di0].y));
            if (std::abs(q_dist - d_dist) < pair_dist_tol)
                di1_cands.push_back(dj);
        }
        if (di1_cands.empty()) continue;

        std::uniform_int_distribution<int> d1_dist(0, static_cast<int>(di1_cands.size()) - 1);
        int di1 = di1_cands[d1_dist(rng)];

        double cos_t, sin_t, tx, ty;
        bool ok = estimate_rigid_2d(
            query.trunks[qi0].x, query.trunks[qi0].y,
            db.trunks[di0].x, db.trunks[di0].y,
            query.trunks[qi1].x, query.trunks[qi1].y,
            db.trunks[di1].x, db.trunks[di1].y,
            cos_t, sin_t, tx, ty);
        if (!ok) continue;

        double score = compute_forward_score(
            query, db, cos_t, sin_t, tx, ty,
            thresh_sq, params);

        if (score > best_score)
            best_score = score;
    }

    return best_score;
}
