// Inter-Sequence Evaluation: ForestLPR Table 2 protocol
// Usage: ./trunk_lpr inter <config.yaml>

#include "inter_eval.hpp"
#include "types.hpp"
#include "trunk_extractor.hpp"
#include "spectrum_descriptor.hpp"
#include "local_matcher.hpp"
#include "pose_loader.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <atomic>
#include <yaml-cpp/yaml.h>

// GT loading
struct QueryEntry {
    int frame_idx;                // query frame index (= scan_id)
    std::vector<int> positives;   // list of positive DB indices
};

struct PairGT {
    int query_seq, db_seq;
    std::vector<QueryEntry> queries;
};

struct InterGT {
    int num_seq;
    std::vector<int> db_frame_counts;
    std::vector<PairGT> pairs;    // query->db sequence pairs (size depends on num_seq)
};

InterGT load_inter_gt(const std::string& path) {
    InterGT gt;
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::cerr << "[ERROR] Failed to open GT file: " << path << std::endl;
        return gt;
    }

    ifs >> gt.num_seq;
    gt.db_frame_counts.resize(gt.num_seq);
    for (int i = 0; i < gt.num_seq; i++)
        ifs >> gt.db_frame_counts[i];

    std::string token;
    while (ifs >> token) {
        if (token != "PAIR") continue;
        PairGT pair;
        int num_queries;
        ifs >> pair.query_seq >> pair.db_seq >> num_queries;
        pair.queries.resize(num_queries);
        for (int qi = 0; qi < num_queries; qi++) {
            int frame_idx, num_pos;
            ifs >> frame_idx >> num_pos;
            pair.queries[qi].frame_idx = frame_idx;
            pair.queries[qi].positives.resize(num_pos);
            for (int pi = 0; pi < num_pos; pi++)
                ifs >> pair.queries[qi].positives[pi];
        }
        gt.pairs.push_back(std::move(pair));
    }
    return gt;
}

// Inter-Sequence
int run_inter(const std::string& config_path) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::cout << std::unitbuf;

    YAML::Node config;
    try { config = YAML::LoadFile(config_path); }
    catch (const YAML::Exception& e) {
        std::cerr << "[ERROR] Failed to parse YAML: " << e.what() << std::endl;
        return 1;
    }

    std::string env_name = config["environment"].as<std::string>();
    std::string gt_file = config["gt_file"].as<std::string>();
    auto sequences = config["sequences"];
    int num_seq = static_cast<int>(sequences.size());

    std::cout << "=== Inter-Sequence Evaluation: " << env_name << " ===" << std::endl;

    // Load GT
    auto gt = load_inter_gt(gt_file);
    if (gt.pairs.empty()) {
        std::cerr << "[ERROR] Failed to load GT" << std::endl;
        return 1;
    }
    std::cout << "  GT: " << gt.pairs.size() << " pairs" << std::endl;

    // Guard: the number of config sequences must match GT num_seq (prevents out-of-bounds all_scans/seq indexing)
    if (gt.num_seq != num_seq) {
        std::cerr << "[ERROR] config sequences(" << num_seq << ") != GT num_seq("
                  << gt.num_seq << ")" << std::endl;
        return 1;
    }
    for (const auto& gp : gt.pairs) {
        if (gp.query_seq < 0 || gp.query_seq >= num_seq ||
            gp.db_seq < 0 || gp.db_seq >= num_seq) {
            std::cerr << "[ERROR] GT pair sequence index out of range: query_seq=" << gp.query_seq
                      << ", db_seq=" << gp.db_seq << " (num_seq=" << num_seq << ")" << std::endl;
            return 1;
        }
    }

    // Load sequences
    std::vector<std::string> seq_names(num_seq);
    std::vector<std::vector<ScanData>> all_scans(num_seq);

    auto trunk_params = load_trunk_params(config);
    auto spec_params = load_spectrum_params(config);

    auto total_t0 = std::chrono::high_resolution_clock::now();

    for (int si = 0; si < num_seq; si++) {
        seq_names[si] = sequences[si]["name"].as<std::string>();
        std::string pcd_folder = sequences[si]["pcd_folder"].as<std::string>();
        std::string poses_file = sequences[si]["poses_file"].as<std::string>();

        all_scans[si] = load_poses(poses_file, pcd_folder);
        int n_scans = static_cast<int>(all_scans[si].size());

        // Trunk extraction
        std::atomic<int> prog(0);
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n_scans; i++) {
            all_scans[si][i].trunks = extract_trunks(all_scans[si][i].file_path, trunk_params);
            int done = ++prog;
            if (done % 500 == 0 || done == n_scans) {
                #pragma omp critical
                std::cout << "  " << seq_names[si] << " trunk: " << done << "/" << n_scans << std::endl;
            }
        }
        // Spectrum
        for (auto& s : all_scans[si])
            compute_spectrum(s, spec_params);

        double avg_t = 0;
        for (auto& s : all_scans[si]) avg_t += s.trunks.size();
        avg_t /= n_scans;
        std::cout << "  " << seq_names[si] << ": " << n_scans
                  << " scans, mean trunks=" << std::setprecision(1) << std::fixed
                  << avg_t << std::endl;

        // Validate the number of DB frames (prevents out-of-bounds frame_idx; aborts on mismatch)
        int actual_n = static_cast<int>(all_scans[si].size());
        if (gt.db_frame_counts[si] != actual_n) {
            std::cerr << "[ERROR] " << seq_names[si] << " frame count mismatch: GT="
                      << gt.db_frame_counts[si] << " actual=" << actual_n << ", aborting" << std::endl;
            return 1;
        }
    }

    // Evaluation
    auto bp = load_ransac_params(config);
    auto ev_node = config["evaluation"];
    int top_n = ev_node["spectrum_top_n"] ? ev_node["spectrum_top_n"].as<int>() : 50;  // same fallback as intra (load_eval_params)

    std::cout << "\n=== RANSAC parameters ===" << std::endl;
    std::cout << "  Spectrum top-" << top_n
              << ", iters=" << bp.ransac_iters
              << std::setprecision(2) << std::fixed
              << ", inlier_thresh=" << bp.inlier_thresh
              << ", dcr=" << bp.density_compat_ratio
              << ", sigma_d=" << bp.density_sigma_d << std::endl;

    // R@1 Grid (num_seq × num_seq)
    std::vector<std::vector<double>> r1_grid(num_seq, std::vector<double>(num_seq, -1.0));
    double total_recall_sum = 0;
    int total_pairs = 0;

    auto eval_t0 = std::chrono::high_resolution_clock::now();

    for (auto& pair : gt.pairs) {
        int qsi = pair.query_seq;
        int dsi = pair.db_seq;
        auto& q_scans = all_scans[qsi];
        auto& d_scans = all_scans[dsi];
        int n_db = static_cast<int>(d_scans.size());
        int n_queries = static_cast<int>(pair.queries.size());

        auto pair_t0 = std::chrono::high_resolution_clock::now();

        std::atomic<int> num_evaluated(0), num_correct(0), progress(0);

        #pragma omp parallel for schedule(dynamic)
        for (int qi = 0; qi < n_queries; qi++) {
            auto& qe = pair.queries[qi];
            if (qe.positives.empty()) {
                ++progress;
                continue;
            }
            if (qe.frame_idx < 0 || qe.frame_idx >= static_cast<int>(q_scans.size())) {
                ++progress;
                continue;  // Prevent out-of-bounds frame_idx (undefined behavior) from a corrupted GT file
            }

            auto& query = q_scans[qe.frame_idx];
            if (query.spectrum.empty()) {
                ++progress;
                continue;
            }

            // Select the Spectrum top-K
            std::vector<std::pair<double, int>> sims;
            sims.reserve(n_db);
            for (int di = 0; di < n_db; di++) {
                if (d_scans[di].spectrum.empty()) continue;
                double sim = spectrum_similarity(query, d_scans[di]);
                sims.push_back({sim, di});
            }

            int actual_top = std::min(top_n, static_cast<int>(sims.size()));
            std::partial_sort(sims.begin(), sims.begin() + actual_top, sims.end(),
                            [](auto& a, auto& b) { return a.first > b.first; });

            num_evaluated++;

            // RANSAC reranking
            {
                double best_score = -1;
                int best_idx = -1;
                for (int k = 0; k < actual_top; k++) {
                    int di = sims[k].second;
                    double score = ransac_score(query, d_scans[di], bp);
                    if (score > best_score) {
                        best_score = score;
                        best_idx = di;
                    }
                }

                // Count as correct if the top-1 candidate is a positive
                if (best_idx >= 0) {
                    for (int p : qe.positives) {
                        if (p == best_idx) {
                            num_correct++;
                            break;
                        }
                    }
                }
            }

            int done = ++progress;
            if (done % 500 == 0 || done == n_queries) {
                #pragma omp critical
                std::cout << "    " << seq_names[qsi] << "→" << seq_names[dsi]
                          << ": " << done << "/" << n_queries << std::endl;
            }
        }

        auto pair_t1 = std::chrono::high_resolution_clock::now();
        double pair_sec = std::chrono::duration<double>(pair_t1 - pair_t0).count();

        int ne = num_evaluated.load(), nc = num_correct.load();
        double recall_1 = (ne > 0) ? (double)nc / ne * 100.0 : 0.0;
        r1_grid[qsi][dsi] = recall_1;
        total_recall_sum += recall_1;
        total_pairs++;

        std::cout << "  " << seq_names[qsi] << "→" << seq_names[dsi]
                  << ": R@1=" << std::setprecision(2) << std::fixed << recall_1
                  << "% (" << nc << "/" << ne << ")"
                  << " [" << std::setprecision(1) << pair_sec << " s]" << std::endl;
    }

    auto eval_t1 = std::chrono::high_resolution_clock::now();
    double eval_sec = std::chrono::duration<double>(eval_t1 - eval_t0).count();
    auto total_t1 = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(total_t1 - total_t0).count();

    // Results
    double ave_r1 = (total_pairs > 0) ? total_recall_sum / total_pairs : 0.0;

    std::cout << "\n=== R@1 Grid (" << env_name << ") ===" << std::endl;
    std::cout << std::setw(8) << "Q\\DB";
    for (int i = 0; i < num_seq; i++)
        std::cout << std::setw(10) << seq_names[i];
    std::cout << std::endl;
    for (int i = 0; i < num_seq; i++) {
        std::cout << std::setw(8) << seq_names[i];
        for (int j = 0; j < num_seq; j++) {
            if (i == j) std::cout << std::setw(10) << "-";
            else std::cout << std::setw(9) << std::setprecision(2) << std::fixed << r1_grid[i][j] << "%";
        }
        std::cout << std::endl;
    }

    std::cout << "\n=== Inter-" << env_name.substr(0,1) << " R@1 = "
              << std::setprecision(2) << std::fixed << ave_r1 << "% ===" << std::endl;
    std::cout << "  Evaluation time: " << std::setprecision(1) << eval_sec << " s"
              << ", total time: " << std::setprecision(1) << total_sec << " s" << std::endl;

    std::cout << "\n=== Done ===" << std::endl;
    return 0;
}
