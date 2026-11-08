#include "pose_loader.hpp"

#include <fstream>
#include <sstream>
#include <iostream>

std::vector<ScanData> load_poses(const std::string& csv_path,
                                 const std::string& pcd_folder) {
    std::vector<ScanData> scans;
    std::ifstream ifs(csv_path);
    if (!ifs.is_open()) {
        std::cerr << "[ERROR] Cannot open CSV file: " << csv_path << std::endl;
        return scans;
    }

    std::string header;
    std::getline(ifs, header);

    bool has_index = (header.rfind("timestamp", 0) != 0);

    std::string line;
    int scan_id = 0;
    int line_num = 1;
    while (std::getline(ifs, line)) {
        line_num++;
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string token;

        try {
            if (has_index) std::getline(ss, token, ',');
            std::string ts_str;
            std::getline(ss, ts_str, ',');
            double timestamp = std::stod(ts_str);

            std::getline(ss, token, ',');
            double x = std::stod(token);
            std::getline(ss, token, ',');
            double y = std::stod(token);

            ScanData scan;
            scan.scan_id = scan_id++;
            scan.timestamp = timestamp;
            scan.pose_x = x;
            scan.pose_y = y;
            scan.file_path = pcd_folder + ts_str + ".pcd";
            scans.push_back(std::move(scan));
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] CSV line " << line_num << ": parse failed: " << e.what()
                      << " - aborting load to keep scan_id consistent" << std::endl;
            return {};
        }
    }
    // Check that timestamps increase monotonically (the start_time/tt logic assumes sorted input, so unsorted input is rejected explicitly)
    for (size_t i = 1; i < scans.size(); i++) {
        if (scans[i].timestamp < scans[i - 1].timestamp) {
            std::cerr << "[ERROR] poses timestamps are not monotonic (idx " << i << ": " << scans[i].timestamp
                      << " < " << scans[i - 1].timestamp << ") - sorting required, aborting load" << std::endl;
            return {};
        }
    }
    return scans;
}
