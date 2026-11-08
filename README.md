# Trunk-Based LiDAR Place Recognition in Forests Using Local Context Density

C++ implementation of the paper *Trunk-Based LiDAR Place Recognition in Forests Using Local Context Density*.

TrunkLPR extracts tree trunks from a height slice of each LiDAR submap, encodes the point density around each trunk as a Local Context Density, retrieves candidate places with a Trunk Spectrum Descriptor, and selects the final match with density-guided RANSAC. It requires no training and runs on a CPU.

## Requirements

Tested on Ubuntu 22.04 with the following packages.

| Package | Version |
|---|---|
| GCC | 11.4.0 (C++17) |
| CMake | 3.22.1 |
| PCL | 1.12.1 |
| yaml-cpp | 0.7.0 |
| Eigen | 3.4.0 |
| Boost | 1.74 |
| OpenMP | optional, used for parallel execution |

```bash
sudo apt install g++ cmake make libpcl-dev libyaml-cpp-dev
```

If `libpcl-dev` cannot be installed because of a `libusb` version conflict, enable the `jammy-updates` repository and run `sudo apt update`.

## Data

The experiments use the height-normalized submaps (`Clouds_normalized`) and poses (`poses_aligned.csv`) of Wild-Places and BotanicGarden released by [ForestLPR](https://github.com/shenyanqing1105/ForestLPR-CVPR2025) (see *Required Submap Data* in its README). Wild-Places v3.0 regenerated the point clouds and trajectories, so it is not interchangeable with this release.

Place the data at `dataset/` in the repository root, or create a symbolic link with that name. In the ForestLPR release, BotanicGarden is stored one level deeper (e.g., `botanic/1005-03/02/`); place the contents of its 1005-03 and 1005-06 sequence folders in `dataset/botanic/100503/` and `dataset/botanic/100506/`, as shown below.

```
dataset/
├── Wild-Places/
│   ├── Venman/
│   │   └── V-01, V-02, V-03, V-04/
│   │       ├── Clouds_normalized/
│   │       └── poses_aligned.csv
│   └── Karawatha/
│       └── K-01, K-02, K-03, K-04/
│           ├── Clouds_normalized/
│           └── poses_aligned.csv
└── botanic/
    └── 100503, 100506/
        ├── Clouds_normalized/
        └── poses_aligned.csv
```

## Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Run

Run the program from `build/`, because the paths in the configuration files are relative to it.

```bash
# Intra-sequence evaluation
./trunk_lpr intra ../config/intra_K04.yaml

# Inter-sequence evaluation
./trunk_lpr inter ../config/inter_venman.yaml
```

`config/` provides `intra_V03`, `intra_V04`, `intra_K03`, `intra_K04`, `intra_botanic_100503`, `intra_botanic_100506`, `inter_venman`, and `inter_karawatha`.

## Expected results

The RANSAC random seed is derived from the scan IDs, so repeated runs give identical results.

| Intra-sequence | V-03 | V-04 | K-03 | K-04 | Bo.1005-03 | Bo.1005-06 |
|---|---|---|---|---|---|---|
| Recall@1 (%) | 77.18 | 93.49 | 82.26 | 96.85 | 92.44 | 82.09 |
| Maximum F1 (%) | 71.71 | 94.65 | 79.58 | 94.87 | 94.64 | 87.60 |

| Inter-sequence | Inter-V | Inter-K |
|---|---|---|
| Recall@1 (%), mean of 12 pairs | 92.37 | 79.20 |

## Ablation studies

Each ablation changes only the values below in a configuration file.

| Ablation | Section | Change |
|---|---|---|
| w/o Slicing | `trunk_extraction` | `height_slices: [[0.0, 100.0]]` |
| w/o LC Profile | `spectrum` | `bins_lc_sp_3d_profile: 1` |
| w/o LC Radial | `spectrum` | `bins_lc_sp_3d_radial: 1` |
| w/o Spatial Distance | `spectrum` | `bins_lc_sp_3d_spatial: 1` |
| 1D Concatenation | `spectrum` | add `descriptor_concat_1d: true` |
| Vanilla RANSAC | `ransac` | `density_compat_ratio: 0.0`, `density_sigma_d: 0.0`, `use_spread: false` |

## Evaluation on a frame subset

The comparison with BTC, TreeLoc, and TreeLoc++ in the paper uses only the frames for which the TreeLoc++ authors release tree inventories (23-31% of the frames of each sequence). `subset/` contains the poses (`poses/`), the inter-sequence ground truth (`gt/`), and the configuration files (`config/`) for running the proposed method on these frames.

```bash
./trunk_lpr intra ../subset/config/intra_V04_subset.yaml
./trunk_lpr inter ../subset/config/inter_venman_subset.yaml
```

| Frame subset | V-03 | V-04 | K-03 | K-04 | Inter-V | Inter-K |
|---|---|---|---|---|---|---|
| Recall@1 (%) | 73.75 | 94.66 | 81.80 | 98.17 | 89.74 | 79.78 |
| Maximum F1 (%) | 69.91 | 95.12 | 78.45 | 96.57 | - | - |

## Comparison methods

[BTC](https://github.com/hku-mars/btc_descriptor) (commit 742af15), [TreeLoc](https://github.com/minwoo0611/TreeLoc) (28e898e), and [TreeLoc++](https://github.com/minwoo0611/TreeLoc-plusplus) (2fc772e) were evaluated from their public code under the protocol described in the paper. Our evaluation scripts are available from the authors upon reasonable request.

## License

The source code is released under the MIT License (see LICENSE).

The files in `gt/` and `subset/` are derived from the [Wild-Places](https://github.com/csiro-robotics/Wild-Places) dataset (Knights et al., ICRA 2023) and are distributed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The files in `gt/` were converted to text from the evaluation files released by ForestLPR, and the files in `subset/` restrict the poses and ground truth to the frames described above.
