// Intra-Sequence Evaluation — Spectrum top-K → Density-Guided RANSAC
// Usage: ./trunk_lpr intra <config.yaml>

#include "intra_eval.hpp"
#include "types.hpp"
#include "trunk_extractor.hpp"
#include "evaluator.hpp"
#include "spectrum_descriptor.hpp"
#include "local_matcher.hpp"
#include "pose_loader.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <chrono>
#include <atomic>

#include <yaml-cpp/yaml.h>

int run_intra(const std::string& config_path) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::cout << std::unitbuf;

    // Data loading
    std::cout << "=== TrunkLPR (Intra-Sequence) ===" << std::endl;

    YAML::Node config;
    try {
        config = YAML::LoadFile(config_path);
    } catch (const YAML::Exception& e) {
        std::cerr << "[ERROR] Failed to parse YAML: " << e.what() << std::endl;
        return 1;
    }

    std::string pcd_folder = config["data"]["pcd_folder"].as<std::string>();
    std::string poses_file = config["data"]["poses_file"].as<std::string>();
    std::cout << "  PCD folder: " << pcd_folder << std::endl;
    std::cout << "  Poses file: " << poses_file << std::endl;

    auto scans = load_poses(poses_file, pcd_folder);
    std::cout << "  Loaded scans: " << scans.size() << std::endl;
    if (scans.empty()) { std::cerr << "[ERROR] No scans found" << std::endl; return 1; }

    double trunk_sec = 0.0;
    double spec_ms = 0.0;

    // Trunk extraction
    std::cout << "\n=== Trunk extraction ===" << std::endl;
    auto trunk_params = load_trunk_params(config);
    auto trunk_t0 = std::chrono::high_resolution_clock::now();

    int total_scans = static_cast<int>(scans.size());
    std::atomic<int> trunk_progress(0);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < total_scans; i++) {
        scans[i].trunks = extract_trunks(scans[i].file_path, trunk_params);
        int done = ++trunk_progress;
        if (done % 200 == 0 || done == total_scans) {
            #pragma omp critical
            std::cout << "  Progress: " << done << "/" << total_scans << std::endl;
        }
    }

    auto trunk_t1 = std::chrono::high_resolution_clock::now();
    trunk_sec = std::chrono::duration<double>(trunk_t1 - trunk_t0).count();

    int zero_count = 0;
    double avg_trunks = 0;
    for (const auto& s : scans) {
        avg_trunks += s.trunks.size();
        if (s.trunks.empty()) zero_count++;
    }
    avg_trunks /= scans.size();

    std::cout << "  Done (total time: " << std::setprecision(1) << std::fixed << trunk_sec << " s)" << std::endl;
    std::cout << "  Mean trunks: " << std::setprecision(1) << avg_trunks
              << ", trunk=0: " << zero_count << "/" << scans.size() << std::endl;

    // Spectrum Descriptor
    auto spec_params = load_spectrum_params(config);

    auto t2 = std::chrono::high_resolution_clock::now();
    for (auto& s : scans) {
        compute_spectrum(s, spec_params);
    }
    auto t3 = std::chrono::high_resolution_clock::now();

    spec_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    int spec_dim = 0;
    for (const auto& s : scans) {
        if (!s.spectrum.empty()) { spec_dim = static_cast<int>(s.spectrum.size()); break; }
    }
    std::cout << "\n=== Spectrum Descriptor (" << spec_dim << "D) ===" << std::endl;

    int valid_spec = 0;
    for (const auto& s : scans) if (!s.spectrum.empty()) valid_spec++;
    std::cout << "  Valid spectra: " << valid_spec << "/" << scans.size()
              << " (time: " << std::setprecision(1) << spec_ms << "ms)" << std::endl;

    auto eval_params = load_eval_params(config);

    SimilarityFunc global_sim = [](const ScanData& a, const ScanData& b) -> double {
        return spectrum_similarity(a, b);
    };

    // Spectrum top-K -> RANSAC (single run)
    auto bp = load_ransac_params(config);

    auto ransac_fn = [&bp](const ScanData& a, const ScanData& b) -> double {
        return ransac_score(a, b, bp);
    };

    std::cout << "\n=== RANSAC ===" << std::endl;
    std::cout << "  Parameters:" << std::endl;
    std::cout << "    Spectrum top-" << eval_params.spectrum_top_n << std::endl;
    std::cout << "    RANSAC iters=" << bp.ransac_iters << std::endl;
    std::cout << std::setprecision(2) << std::fixed
              << "    inlier_thresh=" << bp.inlier_thresh << "m"
              << ", density_compat=" << bp.density_compat_ratio
              << ", density_sigma_d=" << bp.density_sigma_d << std::endl;

    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = evaluate_rerank(scans, eval_params, global_sim, ransac_fn);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ransac_sec = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "\n  Result: R@1=" << std::setprecision(2) << result.recall_1 * 100.0
              << "% (" << result.num_correct << "/" << result.num_revisits
              << ") F1max=" << std::setprecision(2) << result.f1_max * 100.0 << "%"
              << " (total time: " << std::setprecision(1) << std::fixed << ransac_sec << " s"
              << ", queries: " << result.num_queries << ")" << std::endl;

    std::cout << "\n=== Done ===" << std::endl;
    return 0;
}
