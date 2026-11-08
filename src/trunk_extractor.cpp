#include "trunk_extractor.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <iostream>

TrunkExtractorParams load_trunk_params(const YAML::Node& config) {
    TrunkExtractorParams p;
    auto te = config["trunk_extraction"];

    auto hs = te["height_slices"];
    if (hs.size() > 1)
        std::cerr << "[WARN] height_slices contains " << hs.size()
                  << " slices; using only the first one (single-slice design)" << std::endl;
    p.height_slice = { hs[0][0].as<double>(), hs[0][1].as<double>() };
    p.dbscan_eps = te["dbscan_eps"].as<double>();
    p.dbscan_min_pts = te["dbscan_min_pts"].as<int>();
    if (te["lc_height_bands"]) {
        auto bands = te["lc_height_bands"];
        p.lc_height_bands.clear();
        for (size_t i = 0; i < bands.size(); i++) {
            p.lc_height_bands.emplace_back(bands[i][0].as<double>(), bands[i][1].as<double>());
        }
    }
    if (te["lc_radii"]) {
        auto radii = te["lc_radii"];
        p.lc_radii.clear();
        for (size_t i = 0; i < radii.size(); i++) {
            p.lc_radii.push_back(radii[i].as<double>());
        }
    }

    return p;
}

// Cluster in the height slice: centroid and diameter
struct ClusterInfo {
    double cx, cy;       // centroid (BEV)
    double diameter;     // 2 * repr_r
};

static std::vector<ClusterInfo> cluster_slice(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    double z_min, double z_max,
    double eps, int min_pts)
{
    // 1) Height slicing and XY projection (2D clustering); the original z is kept separately for PCA
    pcl::PointCloud<pcl::PointXYZ>::Ptr slice(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<float> orig_z;
    for (const auto& pt : cloud->points) {
        if (pt.z >= z_min && pt.z <= z_max) {
            pcl::PointXYZ p;
            p.x = pt.x;
            p.y = pt.y;
            p.z = 0.0f;  // z=0 for 2D clustering
            slice->push_back(p);
            orig_z.push_back(pt.z);
        }
    }

    std::vector<ClusterInfo> clusters;
    if (slice->size() < static_cast<size_t>(min_pts)) return clusters;

    // 2) Euclidean Cluster Extraction (≈ DBSCAN)
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(slice);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(eps);
    ec.setMinClusterSize(min_pts);
    ec.setMaxClusterSize(100000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(slice);
    ec.extract(cluster_indices);

    // 3) For each cluster: centroid + cylinder_pca_rms diameter
    for (const auto& indices : cluster_indices) {
        double sum_x = 0, sum_y = 0, sum_z = 0;
        int n = static_cast<int>(indices.indices.size());
        for (int idx : indices.indices) {
            sum_x += slice->points[idx].x;
            sum_y += slice->points[idx].y;
            sum_z += orig_z[idx];
        }
        double cx = sum_x / n;
        double cy = sum_y / n;
        double cz = sum_z / n;

        // 3D PCA: top eigenvector (cylinder axis candidate)
        Eigen::Vector3d cyl_axis(0, 0, 1);  // default: z axis (vertical cylinder)
        if (n >= 3) {
            double cxx = 0, cxy = 0, cxz = 0, cyy = 0, cyz = 0, czz = 0;
            for (int k = 0; k < n; k++) {
                int idx = indices.indices[k];
                double dx = slice->points[idx].x - cx;
                double dy = slice->points[idx].y - cy;
                double dz = orig_z[idx] - cz;
                cxx += dx * dx; cxy += dx * dy; cxz += dx * dz;
                cyy += dy * dy; cyz += dy * dz;
                czz += dz * dz;
            }
            Eigen::Matrix3d cov;
            cov << cxx, cxy, cxz,
                   cxy, cyy, cyz,
                   cxz, cyz, czz;
            cov /= static_cast<double>(n);
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
            Eigen::Vector3d v_max = solver.eigenvectors().col(2);
            cyl_axis = v_max.normalized();
        }

        // Diameter: cylinder_pca_rms, the RMS perpendicular distance to the PCA axis (max-radius fallback if n<3)
        double repr_r;
        if (n < 3) {
            std::vector<double> radii(n);
            for (int k = 0; k < n; k++) {
                int idx = indices.indices[k];
                double dx = slice->points[idx].x - cx;
                double dy = slice->points[idx].y - cy;
                radii[k] = std::sqrt(dx * dx + dy * dy);
            }
            repr_r = *std::max_element(radii.begin(), radii.end());
        } else {
            double sum_sq = 0.0;
            for (int k = 0; k < n; k++) {
                int idx = indices.indices[k];
                Eigen::Vector3d p(slice->points[idx].x - cx,
                                  slice->points[idx].y - cy,
                                  orig_z[idx] - cz);
                double proj = p.dot(cyl_axis);
                Eigen::Vector3d perp = p - proj * cyl_axis;
                sum_sq += perp.squaredNorm();
            }
            repr_r = std::sqrt(sum_sq / n);
        }

        ClusterInfo ci;
        ci.cx = cx;
        ci.cy = cy;
        ci.diameter = 2.0 * repr_r;
        clusters.push_back(ci);
    }

    return clusters;
}

std::vector<Trunk> extract_trunks(const std::string& pcd_path,
                                  const TrunkExtractorParams& params) {
    // Load the PCD
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_path, *cloud) == -1) {
        std::cerr << "[WARN] Failed to load PCD: " << pcd_path << std::endl;
        return {};
    }

    // Single-slice clustering: each cluster is a trunk
    const double z_min = params.height_slice.first;
    const double z_max = params.height_slice.second;
    std::vector<ClusterInfo> clusters = cluster_slice(
        cloud, z_min, z_max, params.dbscan_eps, params.dbscan_min_pts);

    std::vector<Trunk> trunks;
    trunks.reserve(clusters.size());
    for (const auto& c : clusters) {
        Trunk t;
        t.x = c.cx;
        t.y = c.cy;
        t.diameter = c.diameter;
        trunks.push_back(t);
    }

    // Multi-height Local Context Density: NH height bands × NR radii
    if (!trunks.empty()) {
        const int NH = static_cast<int>(params.lc_height_bands.size());
        const int NR = static_cast<int>(params.lc_radii.size());
        const int LC_DIM = NH * NR;

        // Build a BEV cloud and KD-tree for each height band
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> band_clouds(NH);
        std::vector<pcl::search::KdTree<pcl::PointXYZ>::Ptr> band_trees(NH);
        std::vector<bool> band_valid(NH, false);

        for (int h = 0; h < NH; h++) {
            band_clouds[h].reset(new pcl::PointCloud<pcl::PointXYZ>);
            double z_lo = params.lc_height_bands[h].first;
            double z_hi = params.lc_height_bands[h].second;
            for (const auto& pt : cloud->points) {
                if (pt.z >= z_lo && pt.z < z_hi) {
                    pcl::PointXYZ p;
                    p.x = pt.x;
                    p.y = pt.y;
                    p.z = 0.0f;
                    band_clouds[h]->push_back(p);
                }
            }
            if (band_clouds[h]->size() >= 2) {
                band_trees[h].reset(new pcl::search::KdTree<pcl::PointXYZ>);
                band_trees[h]->setInputCloud(band_clouds[h]);
                band_valid[h] = true;
            }
        }

        std::vector<double> r_sq_lc(NR), areas_lc(NR);
        double max_radius = 0.0;
        for (int r = 0; r < NR; r++) {
            r_sq_lc[r] = params.lc_radii[r] * params.lc_radii[r];
            areas_lc[r] = M_PI * r_sq_lc[r];
            if (params.lc_radii[r] > max_radius) max_radius = params.lc_radii[r];
        }

        std::vector<int> lc_indices;
        std::vector<float> lc_sq_dists;
        for (auto& t : trunks) {
            t.lc_num_heights = NH;
            t.lc_num_radii = NR;
            t.local_context.assign(LC_DIM, 0.0);

            pcl::PointXYZ query;
            query.x = static_cast<float>(t.x);
            query.y = static_cast<float>(t.y);
            query.z = 0.0f;

            for (int h = 0; h < NH; h++) {
                if (!band_valid[h]) continue;
                band_trees[h]->radiusSearch(query, max_radius, lc_indices, lc_sq_dists);

                std::vector<int> cnt(NR, 0);
                for (size_t p = 0; p < lc_sq_dists.size(); p++) {
                    double d2 = static_cast<double>(lc_sq_dists[p]);
                    for (int r = 0; r < NR; r++) {
                        if (d2 < r_sq_lc[r]) cnt[r]++;
                    }
                }
                for (int r = 0; r < NR; r++) {
                    t.local_context[h * NR + r] = static_cast<double>(cnt[r]) / areas_lc[r];
                }
            }
        }
    }

    return trunks;
}
