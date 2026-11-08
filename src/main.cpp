// TrunkLPR: unified entry point
// Usage:  ./trunk_lpr intra <config.yaml>
//         ./trunk_lpr inter <config.yaml>

#include "intra_eval.hpp"
#include "inter_eval.hpp"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <intra|inter> <config.yaml>" << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    std::string config_path = argv[2];

    if (mode == "intra") {
        return run_intra(config_path);
    } else if (mode == "inter") {
        return run_inter(config_path);
    } else {
        std::cerr << "[ERROR] Unknown mode: " << mode << std::endl;
        std::cerr << "Usage: " << argv[0] << " <intra|inter> <config.yaml>" << std::endl;
        return 1;
    }
}
