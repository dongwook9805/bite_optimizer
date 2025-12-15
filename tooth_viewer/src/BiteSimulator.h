#ifndef BITESIMULATOR_H
#define BITESIMULATOR_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>
#include <vector>
#include <unordered_map>
#include "Mesh.h"

struct OrthodonticMetrics {
    // Basic position metrics
    double overjet_mm = 0.0;
    double overbite_mm = 0.0;
    double midline_dev_mm = 0.0;

    // === 제1원칙: 상호 보호 교합 (Mutually Protected Occlusion) ===
    // 구치부가 전체 힘의 90% 이상을 받아야 함
    double force_anterior = 0.0;      // 전치부에 걸리는 힘 (낮을수록 좋음)
    double force_posterior = 0.0;     // 구치부에 걸리는 힘 (높을수록 좋음)
    double protection_ratio = 0.0;    // force_posterior / total (0.9+ 이상이 이상적)

    // === 제2원칙: 좌우 균형 (Bilateral Balance) ===
    double force_left = 0.0;
    double force_right = 0.0;
    double balance_ratio = 0.0;       // 0.5에 가까울수록 좋음 (left / total)
    double balance_error = 0.0;       // |left - right| / total (0에 가까울수록 좋음)

    // === 제3원칙: 치축 방향 부하 (Axial Loading) ===
    // 힘 벡터가 치아 축과 평행해야 함
    double axial_alignment_score = 0.0;  // 1.0에 가까울수록 좋음
    double lateral_force_penalty = 0.0;  // 측방력 페널티 (0에 가까울수록 좋음)

    // === 제4원칙: 접촉 분포 ===
    double contact_evenness = 0.0;    // 접촉 분포 균일도 (분산의 역수)
    int contact_point_count = 0;      // 총 접촉점 수
    double penetration_count = 0;     // 침투(과교합) 점 수

    // Legacy (backwards compatibility)
    double anterior_contact_ratio = 0.0;
    double posterior_contact_ratio = 0.0;
    double left_contact_force = 0.0;
    double right_contact_force = 0.0;
};

// Contact point for visualization (legacy)
struct ContactPoint {
    Eigen::Vector3f position;
    float signedDistance;
    enum Type { GOOD_CONTACT, PENETRATION, GAP } type;
};

// Spatial hash for fast nearest neighbor queries
class SpatialHash {
public:
    void build(const std::vector<Eigen::Vector3f>& points, float cellSize);
    int findNearest(const Eigen::Vector3f& query) const;
    void findInRadius(const Eigen::Vector3f& query, float radius,
                      std::vector<int>& indices) const;

private:
    struct Cell {
        std::vector<int> indices;
    };

    int64_t hashKey(int x, int y, int z) const;
    void getCellCoord(const Eigen::Vector3f& p, int& x, int& y, int& z) const;

    std::unordered_map<int64_t, Cell> m_cells;
    std::vector<Eigen::Vector3f> m_points;
    float m_cellSize = 1.0f;
    Eigen::Vector3f m_minBound;
};

class BiteSimulator {
public:
    BiteSimulator();
    ~BiteSimulator();

    // Load meshes
    bool loadMaxilla(const std::string& path);
    bool loadMandible(const std::string& path);

    // Get meshes
    Mesh* maxilla() const { return m_maxilla.get(); }
    Mesh* mandible() const { return m_mandible.get(); }

    // Transform jaws
    void applyTransform(const Eigen::Vector3f& deltaRotationDeg, const Eigen::Vector3f& deltaTranslation, bool moveMaxilla = false);
    void applyRotationMatrix(const Eigen::Matrix3f& rotation, bool moveMaxilla = false);  // For view-relative rotation
    void setMandibleTransform(const Eigen::Matrix4f& transform);

    // Reset to initial state
    void reset();

    // Rough alignment - position mandible below maxilla
    void roughAlign();

    // Landmark-based alignment
    void alignFromLandmarks(const std::vector<Eigen::Vector3f>& maxillaPoints,
                            const std::vector<Eigen::Vector3f>& mandiblePoints);

    // Get current state
    Eigen::Vector3f currentTranslation() const { return m_currentTranslation; }
    Eigen::Vector3f currentRotation() const { return m_currentRotationDeg; }
    Eigen::Matrix4f transformMatrix() const { return m_transformMatrix; }

    // Compute metrics
    OrthodonticMetrics computeMetrics() const;
    double computeReward(const OrthodonticMetrics& metrics) const;

    // Contact visualization - colors mandible vertices
    // Returns RGB colors for each mandible vertex
    std::vector<Eigen::Vector3f> computeContactColors(float maxDist = 1.0f) const;

    // Legacy contact points (kept for compatibility)
    std::vector<ContactPoint> computeContactPoints(float contactThreshold = 0.5f,
                                                    float penetrationThreshold = 0.0f) const;

    // ICP alignment
    void runICPAlignment(int iterations = 30);

    // Simple gradient-based optimization
    void optimizeStep(double learningRate = 0.01);

    // Save optimized mandible
    bool saveMandible(const std::string& path) const;

    // Build spatial hash for maxilla (public for external mesh manipulation)
    void buildMaxillaIndex();

private:
    // Compute signed distances from mandible to maxilla
    std::vector<double> computeSignedDistances() const;

    // Find closest points on maxilla (uses spatial hash)
    void findClosestPoints(const std::vector<Eigen::Vector3f>& queryPoints,
                           std::vector<Eigen::Vector3f>& closestPoints,
                           std::vector<double>& distances) const;

    std::unique_ptr<Mesh> m_maxilla;
    std::unique_ptr<Mesh> m_mandible;
    std::unique_ptr<Mesh> m_mandibleInitial;

    // Current transform state
    Eigen::Vector3f m_currentTranslation = Eigen::Vector3f::Zero();
    Eigen::Vector3f m_currentRotationDeg = Eigen::Vector3f::Zero();
    Eigen::Matrix4f m_transformMatrix = Eigen::Matrix4f::Identity();

    // Pivot point for rotation (mandible centroid)
    Eigen::Vector3f m_pivotPoint = Eigen::Vector3f::Zero();

    // Maxilla data for proximity queries
    std::vector<Eigen::Vector3f> m_maxillaVertices;
    std::vector<Eigen::Vector3f> m_maxillaNormals;
    SpatialHash m_spatialHash;
};

#endif // BITESIMULATOR_H
