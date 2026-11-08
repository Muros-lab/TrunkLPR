#include "evaluator.hpp"

#include <algorithm>
#include <vector>

EvalParams load_eval_params(const YAML::Node& config) {
    EvalParams p;
    auto ev = config["evaluation"];
    p.world_thresh = ev["world_thresh"].as<double>();
    p.time_thresh = ev["time_thresh"].as<double>();
    p.spectrum_top_n = ev["spectrum_top_n"] ? ev["spectrum_top_n"].as<int>() : 50;
    return p;
}

// Spectrum top-K → RANSAC Reranking
EvalResult evaluate_rerank(const std::vector<ScanData>& scans,
                            const EvalParams& eval_params,
                            SimilarityFunc global_sim_fn,
                            SimilarityFunc ransac_fn) {
    EvalResult result = {};

    int N = static_cast<int>(scans.size());
    double start_time = scans.front().timestamp;
    double world_thresh = eval_params.world_thresh;
    double time_thresh = eval_params.time_thresh;

    const int num_thresholds = 1000;  // number of thresholds in the F1max sweep (hard-coded)
    std::vector<double> thresholds(num_thresholds);
    for (int i = 0; i < num_thresholds; i++) {
        thresholds[i] = static_cast<double>(i) / (num_thresholds - 1);  // [0,1] linspace, as in the ForestLPR evaluation code
    }

    // Precompute the list of valid queries (non-revisit queries are included, as in the ForestLPR protocol)
    struct QueryInfo { int qi; int tt; bool revisit; };
    std::vector<QueryInfo> queries;
    int num_revisits = 0;
    for (int qi = 0; qi < N; qi++) {
        double q_time = scans[qi].timestamp;
        if ((q_time - start_time - time_thresh) < 0) continue;
        // Same as the ForestLPR evaluation code: first index in all timestamps whose value is > (q_time - time_thresh)
        int tt = qi;  // fallback (q itself; the loop always breaks because q_time > q_time - time_thresh)
        for (int j = 0; j < N; j++) {
            if (scans[j].timestamp > (q_time - time_thresh)) { tt = j; break; }
        }
        bool revisit = false;
        for (int j = 0; j <= tt; j++) {
            double wd = euclidean_2d(scans[qi].pose_x, scans[qi].pose_y,
                                     scans[j].pose_x, scans[j].pose_y);
            if (wd < world_thresh) { revisit = true; break; }
        }
        if (revisit) num_revisits++;
        queries.push_back({qi, tt, revisit});
    }

    int num_queries = static_cast<int>(queries.size());

    struct QueryResult { bool revisit; double top1_world_dist; double top1_embed_dist; };
    std::vector<QueryResult> qresults(num_queries);

    int spec_top = eval_params.spectrum_top_n;

    #pragma omp parallel for schedule(dynamic)
    for (int qidx = 0; qidx < num_queries; qidx++) {
        auto& qinfo = queries[qidx];
        int qi = qinfo.qi;
        int tt = qinfo.tt;

        // Sort by global score
        struct Candidate { int idx; double global_score; };
        std::vector<Candidate> candidates;
        candidates.reserve(tt + 1);
        for (int j = 0; j <= tt; j++) {
            double gsim = global_sim_fn(scans[qi], scans[j]);
            candidates.push_back({j, gsim});
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b) {
                      return a.global_score > b.global_score;
                  });

        // Spectrum top-K filter
        int all_n = static_cast<int>(candidates.size());
        if (all_n == 0) {
            qresults[qidx] = {qinfo.revisit, 1e9, 1.0};
            continue;
        }
        if (spec_top > 0 && all_n > spec_top) {
            candidates.resize(spec_top);
            all_n = spec_top;
        }

        // RANSAC
        double best_ransac = -1.0;
        int best_idx = candidates[0].idx;

        for (int ci = 0; ci < all_n; ci++) {
            double rs = ransac_fn(scans[qi], scans[candidates[ci].idx]);
            if (rs > best_ransac) {
                best_ransac = rs;
                best_idx = candidates[ci].idx;
            }
        }

        double top1_wd = euclidean_2d(scans[qi].pose_x, scans[qi].pose_y,
                                      scans[best_idx].pose_x, scans[best_idx].pose_y);
        double top1_ed = (best_ransac >= 0) ? 1.0 - best_ransac : 1.0;
        qresults[qidx] = {qinfo.revisit, top1_wd, top1_ed};
    }

    // Aggregate the results (ForestLPR protocol)
    std::vector<int> num_tp(num_thresholds, 0);
    std::vector<int> num_fp(num_thresholds, 0);
    std::vector<int> num_fn(num_thresholds, 0);
    int num_correct = 0;

    for (int qidx = 0; qidx < num_queries; qidx++) {
        auto& qr = qresults[qidx];
        if (qr.revisit && qr.top1_world_dist < world_thresh) num_correct++;

        for (int ti = 0; ti < num_thresholds; ti++) {
            if (qr.top1_embed_dist < thresholds[ti]) {
                if (qr.top1_world_dist < world_thresh) num_tp[ti]++;
                else num_fp[ti]++;
            } else if (qr.revisit) {
                num_fn[ti]++;
            }
        }
    }

    result.num_revisits = num_revisits;
    result.num_correct = num_correct;
    result.num_queries = num_queries;
    result.recall_1 = (num_revisits > 0) ?
        static_cast<double>(num_correct) / num_revisits : 0.0;

    result.f1_max = 0.0;
    for (int ti = 0; ti < num_thresholds; ti++) {
        double tp = num_tp[ti], fp = num_fp[ti];
        double fn = num_fn[ti];
        if (tp > 0) {
            double precision = tp / (tp + fp);
            double recall = tp / (tp + fn);
            double f1 = 2.0 * precision * recall / (precision + recall);
            if (f1 > result.f1_max) result.f1_max = f1;
        }
    }

    return result;
}

