#pragma once

#include "types.hpp"
#include <string>
#include <vector>

// Parse poses_aligned.csv
std::vector<ScanData> load_poses(const std::string& csv_path,
                                 const std::string& pcd_folder);
