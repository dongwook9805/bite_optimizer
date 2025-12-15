#include "BiteSimulator.h"
#include "MeshLoader.h"
#include <cmath>
#include <algorithm>
#include <iostream>

// ========== SpatialHash Implementation ==========

void SpatialHash::build(const std::vector<Eigen::Vector3f>& points, float cellSize) {
    m_points = points;
    m_cellSize = cellSize;
    m_cells.clear();

    if (points.empty()) return;

    // Find bounding box
    m_minBound = points[0];
    Eigen::Vector3f maxBound = points[0];
    for (const auto& p : points) {
        m_minBound = m_minBound.cwiseMin(p);
        maxBound = maxBound.cwiseMax(p);
    }

    // Insert points into cells
    for (size_t i = 0; i < points.size(); ++i) {
        int x, y, z;
        getCellCoord(points[i], x, y, z);
        m_cells[hashKey(x, y, z)].indices.push_back(static_cast<int>(i));
    }
}

int64_t SpatialHash::hashKey(int x, int y, int z) const {
    // Simple spatial hash function
    return static_cast<int64_t>(x) * 73856093LL ^
           static_cast<int64_t>(y) * 19349663LL ^
           static_cast<int64_t>(z) * 83492791LL;
}

void SpatialHash::getCellCoord(const Eigen::Vector3f& p, int& x, int& y, int& z) const {
    x = static_cast<int>(std::floor((p.x() - m_minBound.x()) / m_cellSize));
    y = static_cast<int>(std::floor((p.y() - m_minBound.y()) / m_cellSize));
    z = static_cast<int>(std::floor((p.z() - m_minBound.z()) / m_cellSize));
}

int SpatialHash::findNearest(const Eigen::Vector3f& query) const {
    if (m_points.empty()) return -1;

    int cx, cy, cz;
    getCellCoord(query, cx, cy, cz);

    int bestIdx = -1;
    float bestDistSq = std::numeric_limits<float>::max();

    // Search in expanding radius
    for (int radius = 0; radius <= 3; ++radius) {
        for (int dx = -radius; dx <= radius; ++dx) {
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dz = -radius; dz <= radius; ++dz) {
                    // Only check cells at the current radius boundary
                    if (radius > 0 && std::abs(dx) < radius && std::abs(dy) < radius && std::abs(dz) < radius)
                        continue;

                    auto it = m_cells.find(hashKey(cx + dx, cy + dy, cz + dz));
                    if (it != m_cells.end()) {
                        for (int idx : it->second.indices) {
                            float distSq = (query - m_points[idx]).squaredNorm();
                            if (distSq < bestDistSq) {
                                bestDistSq = distSq;
                                bestIdx = idx;
                            }
                        }
                    }
                }
            }
        }

        // If found something and distance is less than cell diagonal, we're done
        if (bestIdx >= 0 && bestDistSq < (radius + 1) * (radius + 1) * m_cellSize * m_cellSize * 3)
            break;
    }

    // Fallback to brute force if nothing found (shouldn't happen often)
    if (bestIdx < 0) {
        for (size_t i = 0; i < m_points.size(); ++i) {
            float distSq = (query - m_points[i]).squaredNorm();
            if (distSq < bestDistSq) {
                bestDistSq = distSq;
                bestIdx = static_cast<int>(i);
            }
        }
    }

    return bestIdx;
}

void SpatialHash::findInRadius(const Eigen::Vector3f& query, float radius,
                                std::vector<int>& indices) const {
    indices.clear();
    if (m_points.empty()) return;

    int cx, cy, cz;
    getCellCoord(query, cx, cy, cz);

    int cellRadius = static_cast<int>(std::ceil(radius / m_cellSize));
    float radiusSq = radius * radius;

    for (int dx = -cellRadius; dx <= cellRadius; ++dx) {
        for (int dy = -cellRadius; dy <= cellRadius; ++dy) {
            for (int dz = -cellRadius; dz <= cellRadius; ++dz) {
                auto it = m_cells.find(hashKey(cx + dx, cy + dy, cz + dz));
                if (it != m_cells.end()) {
                    for (int idx : it->second.indices) {
                        if ((query - m_points[idx]).squaredNorm() <= radiusSq) {
                            indices.push_back(idx);
                        }
                    }
                }
            }
        }
    }
}

// ========== BiteSimulator Implementation ==========

BiteSimulator::BiteSimulator() {}

BiteSimulator::~BiteSimulator() {}

bool BiteSimulator::loadMaxilla(const std::string& path) {
    m_maxilla = MeshLoader::load(path);
    if (m_maxilla) {
        buildMaxillaIndex();
        return true;
    }
    return false;
}

bool BiteSimulator::loadMandible(const std::string& path) {
    m_mandible = MeshLoader::load(path);
    if (m_mandible) {
        m_mandibleInitial = MeshLoader::load(path);

        Eigen::Vector3f sum = Eigen::Vector3f::Zero();
        for (const auto& v : m_mandible->vertices()) {
            sum += v.position;
        }
        m_pivotPoint = sum / static_cast<float>(m_mandible->vertexCount());

        return true;
    }
    return false;
}

void BiteSimulator::buildMaxillaIndex() {
    if (!m_maxilla) return;

    m_maxillaVertices.clear();
    m_maxillaNormals.clear();

    const auto& vertices = m_maxilla->vertices();
    m_maxillaVertices.reserve(vertices.size());
    m_maxillaNormals.reserve(vertices.size());

    for (const auto& v : vertices) {
        m_maxillaVertices.push_back(v.position);
        m_maxillaNormals.push_back(v.normal);
    }

    // Build spatial hash with cell size based on mesh extent
    Eigen::Vector3f minP = m_maxillaVertices[0];
    Eigen::Vector3f maxP = m_maxillaVertices[0];
    for (const auto& p : m_maxillaVertices) {
        minP = minP.cwiseMin(p);
        maxP = maxP.cwiseMax(p);
    }
    float extent = (maxP - minP).norm();
    float cellSize = extent / 50.0f;  // ~50 cells along longest dimension

    m_spatialHash.build(m_maxillaVertices, cellSize);
}

void BiteSimulator::reset() {
    if (m_mandibleInitial) {
        const auto& initVerts = m_mandibleInitial->vertices();
        auto& verts = const_cast<std::vector<Vertex>&>(m_mandible->vertices());

        for (size_t i = 0; i < verts.size() && i < initVerts.size(); ++i) {
            verts[i].position = initVerts[i].position;
            verts[i].normal = initVerts[i].normal;
        }
    }

    m_currentTranslation = Eigen::Vector3f::Zero();
    m_currentRotationDeg = Eigen::Vector3f::Zero();
    m_transformMatrix = Eigen::Matrix4f::Identity();
}

void BiteSimulator::roughAlign() {
    if (!m_maxilla || !m_mandible) return;

    std::cout << "=== Rough Alignment ===" << std::endl;

    // Step 1: Calculate centroids and average normals
    Eigen::Vector3f maxillaCentroid = Eigen::Vector3f::Zero();
    Eigen::Vector3f maxillaAvgNormal = Eigen::Vector3f::Zero();
    for (const auto& v : m_maxilla->vertices()) {
        maxillaCentroid += v.position;
        maxillaAvgNormal += v.normal;
    }
    maxillaCentroid /= static_cast<float>(m_maxilla->vertexCount());
    maxillaAvgNormal.normalize();

    Eigen::Vector3f mandibleCentroid = Eigen::Vector3f::Zero();
    Eigen::Vector3f mandibleAvgNormal = Eigen::Vector3f::Zero();
    for (const auto& v : m_mandible->vertices()) {
        mandibleCentroid += v.position;
        mandibleAvgNormal += v.normal;
    }
    mandibleCentroid /= static_cast<float>(m_mandible->vertexCount());
    mandibleAvgNormal.normalize();

    std::cout << "  Maxilla centroid: (" << maxillaCentroid.x() << ", " << maxillaCentroid.y() << ", " << maxillaCentroid.z() << ")" << std::endl;
    std::cout << "  Mandible centroid: (" << mandibleCentroid.x() << ", " << mandibleCentroid.y() << ", " << mandibleCentroid.z() << ")" << std::endl;
    std::cout << "  Maxilla avg normal: (" << maxillaAvgNormal.x() << ", " << maxillaAvgNormal.y() << ", " << maxillaAvgNormal.z() << ")" << std::endl;
    std::cout << "  Mandible avg normal: (" << mandibleAvgNormal.x() << ", " << mandibleAvgNormal.y() << ", " << mandibleAvgNormal.z() << ")" << std::endl;

    // Step 2: Rotate mandible so its normal faces opposite to maxilla's normal
    // Target: mandible normal should be opposite to maxilla normal (teeth facing each other)
    Eigen::Vector3f targetNormal = -maxillaAvgNormal;

    // Calculate rotation to align mandible normal to target normal
    Eigen::Vector3f rotationAxis = mandibleAvgNormal.cross(targetNormal);
    float dotProduct = mandibleAvgNormal.dot(targetNormal);

    Eigen::Matrix3f rotation = Eigen::Matrix3f::Identity();

    if (rotationAxis.norm() > 0.001f) {
        rotationAxis.normalize();
        float angle = std::acos(std::clamp(dotProduct, -1.0f, 1.0f));

        // Rodrigues' rotation formula
        Eigen::Matrix3f K;
        K << 0, -rotationAxis.z(), rotationAxis.y(),
             rotationAxis.z(), 0, -rotationAxis.x(),
             -rotationAxis.y(), rotationAxis.x(), 0;

        rotation = Eigen::Matrix3f::Identity() + std::sin(angle) * K + (1 - std::cos(angle)) * K * K;

        std::cout << "  Rotation angle: " << (angle * 180.0f / M_PI) << " degrees" << std::endl;
        std::cout << "  Rotation axis: (" << rotationAxis.x() << ", " << rotationAxis.y() << ", " << rotationAxis.z() << ")" << std::endl;
    } else if (dotProduct < 0) {
        // Normals are already opposite - no rotation needed
        std::cout << "  Normals already opposite - no rotation needed" << std::endl;
    } else {
        // Normals are same direction - need 180 degree rotation
        // Find any perpendicular axis
        Eigen::Vector3f perpAxis = (std::abs(mandibleAvgNormal.x()) < 0.9f) ?
            Eigen::Vector3f(1, 0, 0) : Eigen::Vector3f(0, 1, 0);
        rotationAxis = mandibleAvgNormal.cross(perpAxis).normalized();
        rotation = Eigen::AngleAxisf(M_PI, rotationAxis).toRotationMatrix();
        std::cout << "  180 degree rotation needed" << std::endl;
    }

    // Apply rotation around mandible centroid
    auto& verts = const_cast<std::vector<Vertex>&>(m_mandible->vertices());
    for (auto& v : verts) {
        v.position = rotation * (v.position - mandibleCentroid) + mandibleCentroid;
        v.normal = rotation * v.normal;
    }

    // Also update initial mandible
    if (m_mandibleInitial) {
        auto& initVerts = const_cast<std::vector<Vertex>&>(m_mandibleInitial->vertices());
        for (auto& v : initVerts) {
            v.position = rotation * (v.position - mandibleCentroid) + mandibleCentroid;
            v.normal = rotation * v.normal;
        }
    }

    // Recalculate mandible bounds after rotation
    Eigen::Vector3f mandibleMin(std::numeric_limits<float>::max(),
                                std::numeric_limits<float>::max(),
                                std::numeric_limits<float>::max());
    Eigen::Vector3f mandibleMax(std::numeric_limits<float>::lowest(),
                                std::numeric_limits<float>::lowest(),
                                std::numeric_limits<float>::lowest());
    Eigen::Vector3f newMandibleCentroid = Eigen::Vector3f::Zero();

    for (const auto& v : m_mandible->vertices()) {
        mandibleMin = mandibleMin.cwiseMin(v.position);
        mandibleMax = mandibleMax.cwiseMax(v.position);
        newMandibleCentroid += v.position;
    }
    newMandibleCentroid /= static_cast<float>(m_mandible->vertexCount());

    // Calculate maxilla bounds
    Eigen::Vector3f maxillaMin(std::numeric_limits<float>::max(),
                               std::numeric_limits<float>::max(),
                               std::numeric_limits<float>::max());
    Eigen::Vector3f maxillaMax(std::numeric_limits<float>::lowest(),
                               std::numeric_limits<float>::lowest(),
                               std::numeric_limits<float>::lowest());

    for (const auto& v : m_maxilla->vertices()) {
        maxillaMin = maxillaMin.cwiseMin(v.position);
        maxillaMax = maxillaMax.cwiseMax(v.position);
    }

    Eigen::Vector3f maxillaCenter = (maxillaMin + maxillaMax) * 0.5f;
    Eigen::Vector3f mandibleCenter = (mandibleMin + mandibleMax) * 0.5f;

    // Step 3: Determine vertical axis from maxilla normal direction
    int verticalAxis = 0;
    float maxNormalComp = std::abs(maxillaAvgNormal.x());
    if (std::abs(maxillaAvgNormal.y()) > maxNormalComp) {
        verticalAxis = 1;
        maxNormalComp = std::abs(maxillaAvgNormal.y());
    }
    if (std::abs(maxillaAvgNormal.z()) > maxNormalComp) {
        verticalAxis = 2;
    }

    std::cout << "  Vertical axis: " << (verticalAxis == 0 ? "X" : (verticalAxis == 1 ? "Y" : "Z")) << std::endl;

    // Step 4: Translate to align centers and position with gap
    Eigen::Vector3f translation = Eigen::Vector3f::Zero();

    // Align horizontally (non-vertical axes)
    for (int i = 0; i < 3; ++i) {
        if (i != verticalAxis) {
            translation[i] = maxillaCenter[i] - mandibleCenter[i];
        }
    }

    // Position vertically with small gap
    float gap = 3.0f; // mm

    // Maxilla teeth point in maxillaAvgNormal direction
    // If maxilla normal points negative on vertical axis, maxilla is "above"
    bool maxillaAbove = maxillaAvgNormal[verticalAxis] < 0;

    if (maxillaAbove) {
        // Maxilla above: mandible's max should be just below maxilla's min
        translation[verticalAxis] = maxillaMin[verticalAxis] - mandibleMax[verticalAxis] + gap;
    } else {
        // Maxilla below: mandible's min should be just above maxilla's max
        translation[verticalAxis] = maxillaMax[verticalAxis] - mandibleMin[verticalAxis] - gap;
    }

    std::cout << "  Maxilla above: " << (maxillaAbove ? "yes" : "no") << std::endl;
    std::cout << "  Translation: (" << translation.x() << ", " << translation.y() << ", " << translation.z() << ")" << std::endl;

    // Apply translation
    for (auto& v : verts) {
        v.position += translation;
    }

    if (m_mandibleInitial) {
        auto& initVerts = const_cast<std::vector<Vertex>&>(m_mandibleInitial->vertices());
        for (auto& v : initVerts) {
            v.position += translation;
        }
    }

    // Update pivot point
    m_pivotPoint = newMandibleCentroid + translation;

    // Update transform state
    m_currentTranslation += translation;
    Eigen::Matrix4f transMat = Eigen::Matrix4f::Identity();
    transMat.block<3,3>(0,0) = rotation;
    transMat.block<3,1>(0,3) = translation;
    m_transformMatrix = transMat * m_transformMatrix;

    // Rebuild spatial index
    buildMaxillaIndex();

    std::cout << "  Rough alignment complete" << std::endl;
}

void BiteSimulator::alignFromLandmarks(const std::vector<Eigen::Vector3f>& maxillaPoints,
                                        const std::vector<Eigen::Vector3f>& mandiblePoints) {
    if (!m_mandible || maxillaPoints.size() != mandiblePoints.size() || maxillaPoints.size() < 3) {
        std::cout << "alignFromLandmarks: Invalid input" << std::endl;
        return;
    }

    std::cout << "=== Landmark Alignment ===" << std::endl;
    std::cout << "  Using " << maxillaPoints.size() << " landmark pairs" << std::endl;

    // Compute centroids
    Eigen::Vector3f maxillaCentroid = Eigen::Vector3f::Zero();
    Eigen::Vector3f mandibleCentroid = Eigen::Vector3f::Zero();

    for (size_t i = 0; i < maxillaPoints.size(); ++i) {
        maxillaCentroid += maxillaPoints[i];
        mandibleCentroid += mandiblePoints[i];
    }
    maxillaCentroid /= static_cast<float>(maxillaPoints.size());
    mandibleCentroid /= static_cast<float>(mandiblePoints.size());

    std::cout << "  Maxilla landmark centroid: (" << maxillaCentroid.x() << ", " << maxillaCentroid.y() << ", " << maxillaCentroid.z() << ")" << std::endl;
    std::cout << "  Mandible landmark centroid: (" << mandibleCentroid.x() << ", " << mandibleCentroid.y() << ", " << mandibleCentroid.z() << ")" << std::endl;

    // Build covariance matrix H for SVD
    Eigen::Matrix3f H = Eigen::Matrix3f::Zero();

    for (size_t i = 0; i < maxillaPoints.size(); ++i) {
        Eigen::Vector3f pMaxilla = maxillaPoints[i] - maxillaCentroid;
        Eigen::Vector3f pMandible = mandiblePoints[i] - mandibleCentroid;
        H += pMandible * pMaxilla.transpose();
    }

    // Compute optimal rotation using SVD
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f R = svd.matrixV() * svd.matrixU().transpose();

    // Handle reflection case
    if (R.determinant() < 0) {
        Eigen::Matrix3f V = svd.matrixV();
        V.col(2) *= -1;
        R = V * svd.matrixU().transpose();
        std::cout << "  Corrected reflection in rotation" << std::endl;
    }

    // Compute translation: t = maxillaCentroid - R * mandibleCentroid
    Eigen::Vector3f t = maxillaCentroid - R * mandibleCentroid;

    std::cout << "  Rotation determinant: " << R.determinant() << std::endl;
    std::cout << "  Translation: (" << t.x() << ", " << t.y() << ", " << t.z() << ")" << std::endl;

    // Apply transformation to all mandible vertices
    auto& verts = const_cast<std::vector<Vertex>&>(m_mandible->vertices());
    for (auto& v : verts) {
        v.position = R * v.position + t;
        v.normal = R * v.normal;
    }

    // Also update initial mandible for reset
    if (m_mandibleInitial) {
        auto& initVerts = const_cast<std::vector<Vertex>&>(m_mandibleInitial->vertices());
        for (auto& v : initVerts) {
            v.position = R * v.position + t;
            v.normal = R * v.normal;
        }
    }

    // Update pivot point
    m_pivotPoint = R * m_pivotPoint + t;

    // Update transform state
    m_currentTranslation += t;
    Eigen::Matrix4f transMat = Eigen::Matrix4f::Identity();
    transMat.block<3,3>(0,0) = R;
    transMat.block<3,1>(0,3) = t;
    m_transformMatrix = transMat * m_transformMatrix;

    // Compute alignment error
    float totalError = 0.0f;
    for (size_t i = 0; i < maxillaPoints.size(); ++i) {
        Eigen::Vector3f transformedMandible = R * mandiblePoints[i] + t;
        float error = (transformedMandible - maxillaPoints[i]).norm();
        totalError += error;
        std::cout << "  Pair " << i << " error: " << error << " mm" << std::endl;
    }
    std::cout << "  Average alignment error: " << (totalError / maxillaPoints.size()) << " mm" << std::endl;
    std::cout << "  Landmark alignment complete" << std::endl;
}

void BiteSimulator::applyTransform(const Eigen::Vector3f& deltaRotationDeg,
                                    const Eigen::Vector3f& deltaTranslation,
                                    bool moveMaxilla) {
    Mesh* targetMesh = moveMaxilla ? m_maxilla.get() : m_mandible.get();
    if (!targetMesh) return;

    float rx = deltaRotationDeg.x() * M_PI / 180.0f;
    float ry = deltaRotationDeg.y() * M_PI / 180.0f;
    float rz = deltaRotationDeg.z() * M_PI / 180.0f;

    Eigen::Matrix3f rotX, rotY, rotZ;
    rotX = Eigen::AngleAxisf(rx, Eigen::Vector3f::UnitX());
    rotY = Eigen::AngleAxisf(ry, Eigen::Vector3f::UnitY());
    rotZ = Eigen::AngleAxisf(rz, Eigen::Vector3f::UnitZ());

    Eigen::Matrix3f rotation = rotZ * rotY * rotX;

    // Compute pivot point for the target mesh
    Eigen::Vector3f pivotPoint = Eigen::Vector3f::Zero();
    auto& verts = const_cast<std::vector<Vertex>&>(targetMesh->vertices());
    for (const auto& v : verts) {
        pivotPoint += v.position;
    }
    pivotPoint /= static_cast<float>(verts.size());

    // Apply rotation around pivot and translation
    for (auto& v : verts) {
        Eigen::Vector3f p = v.position - pivotPoint;
        p = rotation * p;
        p += pivotPoint + deltaTranslation;
        v.position = p;
        v.normal = rotation * v.normal;
    }

    // Update state tracking (only for mandible for backwards compatibility)
    if (!moveMaxilla) {
        m_currentTranslation += deltaTranslation;
        m_currentRotationDeg += deltaRotationDeg;

        Eigen::Matrix4f deltaMat = Eigen::Matrix4f::Identity();
        deltaMat.block<3,3>(0,0) = rotation;
        deltaMat.block<3,1>(0,3) = deltaTranslation;
        m_transformMatrix = deltaMat * m_transformMatrix;
    }

    // Rebuild spatial index if maxilla was moved
    if (moveMaxilla) {
        buildMaxillaIndex();
    }
}

void BiteSimulator::applyRotationMatrix(const Eigen::Matrix3f& rotation, bool moveMaxilla) {
    Mesh* targetMesh = moveMaxilla ? m_maxilla.get() : m_mandible.get();
    if (!targetMesh) return;

    // Compute pivot point (centroid of target mesh)
    Eigen::Vector3f pivotPoint = Eigen::Vector3f::Zero();
    auto& verts = const_cast<std::vector<Vertex>&>(targetMesh->vertices());
    for (const auto& v : verts) {
        pivotPoint += v.position;
    }
    pivotPoint /= static_cast<float>(verts.size());

    // Apply rotation around pivot
    for (auto& v : verts) {
        Eigen::Vector3f p = v.position - pivotPoint;
        p = rotation * p;
        v.position = p + pivotPoint;
        v.normal = rotation * v.normal;
    }

    // Rebuild spatial index if maxilla was moved
    if (moveMaxilla) {
        buildMaxillaIndex();
    }
}

void BiteSimulator::setMandibleTransform(const Eigen::Matrix4f& transform) {
    if (!m_mandible || !m_mandibleInitial) return;

    const auto& initVerts = m_mandibleInitial->vertices();
    auto& verts = const_cast<std::vector<Vertex>&>(m_mandible->vertices());

    Eigen::Matrix3f rotation = transform.block<3,3>(0,0);
    Eigen::Vector3f translation = transform.block<3,1>(0,3);

    for (size_t i = 0; i < verts.size() && i < initVerts.size(); ++i) {
        verts[i].position = rotation * initVerts[i].position + translation;
        verts[i].normal = rotation * initVerts[i].normal;
    }

    m_transformMatrix = transform;
}

void BiteSimulator::findClosestPoints(const std::vector<Eigen::Vector3f>& queryPoints,
                                       std::vector<Eigen::Vector3f>& closestPoints,
                                       std::vector<double>& distances) const {
    closestPoints.resize(queryPoints.size());
    distances.resize(queryPoints.size());

    // Use spatial hash for fast lookups
    #pragma omp parallel for
    for (size_t i = 0; i < queryPoints.size(); ++i) {
        int nearestIdx = m_spatialHash.findNearest(queryPoints[i]);
        if (nearestIdx >= 0) {
            closestPoints[i] = m_maxillaVertices[nearestIdx];
            distances[i] = (queryPoints[i] - m_maxillaVertices[nearestIdx]).norm();
        } else {
            closestPoints[i] = queryPoints[i];
            distances[i] = std::numeric_limits<double>::max();
        }
    }
}

std::vector<double> BiteSimulator::computeSignedDistances() const {
    if (!m_mandible || !m_maxilla) return {};

    std::vector<Eigen::Vector3f> mandibleVerts;
    mandibleVerts.reserve(m_mandible->vertexCount());
    for (const auto& v : m_mandible->vertices()) {
        mandibleVerts.push_back(v.position);
    }

    std::vector<Eigen::Vector3f> closestPoints;
    std::vector<double> distances;
    findClosestPoints(mandibleVerts, closestPoints, distances);

    std::vector<double> signedDistances(distances.size());

    #pragma omp parallel for
    for (size_t i = 0; i < distances.size(); ++i) {
        Eigen::Vector3f vec = mandibleVerts[i] - closestPoints[i];

        // Get normal from spatial hash
        int nearestIdx = m_spatialHash.findNearest(closestPoints[i]);
        Eigen::Vector3f normal = (nearestIdx >= 0) ? m_maxillaNormals[nearestIdx] : Eigen::Vector3f::UnitZ();

        double sign = (vec.dot(normal) >= 0) ? 1.0 : -1.0;
        signedDistances[i] = sign * distances[i];
    }

    return signedDistances;
}

std::vector<Eigen::Vector3f> BiteSimulator::computeContactColors(float maxDist) const {
    if (!m_mandible || !m_maxilla) return {};

    const auto& verts = m_mandible->vertices();
    std::vector<Eigen::Vector3f> colors(verts.size());

    // Default mandible color (light blue)
    Eigen::Vector3f defaultColor(0.68f, 0.85f, 0.90f);

    // Get signed distances
    auto signedDists = computeSignedDistances();

    for (size_t i = 0; i < verts.size(); ++i) {
        float dist = static_cast<float>(signedDists[i]);

        if (dist < -0.1f) {
            // Penetration - RED
            float intensity = std::min(1.0f, std::abs(dist) / maxDist);
            colors[i] = Eigen::Vector3f(0.9f, 0.1f + 0.2f * (1.0f - intensity), 0.1f);
        } else if (dist < 0.15f) {
            // Good contact - GREEN
            float intensity = 1.0f - std::abs(dist) / 0.15f;
            colors[i] = Eigen::Vector3f(0.1f + 0.2f * (1.0f - intensity), 0.8f * intensity + 0.3f, 0.1f);
        } else if (dist < maxDist) {
            // Near contact - gradient from yellow to default
            float t = (dist - 0.15f) / (maxDist - 0.15f);
            Eigen::Vector3f yellow(1.0f, 0.8f, 0.2f);
            colors[i] = yellow * (1.0f - t) + defaultColor * t;
        } else {
            // Far - default color
            colors[i] = defaultColor;
        }
    }

    return colors;
}

std::vector<ContactPoint> BiteSimulator::computeContactPoints(float contactThreshold,
                                                               float penetrationThreshold) const {
    std::vector<ContactPoint> contactPoints;
    if (!m_mandible || !m_maxilla) return contactPoints;

    auto signedDists = computeSignedDistances();
    const auto& verts = m_mandible->vertices();

    const int sampleStep = std::max(1, static_cast<int>(verts.size() / 2000));

    for (size_t i = 0; i < verts.size(); i += sampleStep) {
        float signedDist = static_cast<float>(signedDists[i]);

        if (signedDist < contactThreshold && signedDist > -contactThreshold) {
            ContactPoint cp;
            cp.position = verts[i].position;
            cp.signedDistance = signedDist;

            if (signedDist < penetrationThreshold) {
                cp.type = ContactPoint::PENETRATION;
            } else if (signedDist < contactThreshold * 0.3f) {
                cp.type = ContactPoint::GOOD_CONTACT;
            } else {
                cp.type = ContactPoint::GAP;
            }

            contactPoints.push_back(cp);
        }
    }

    return contactPoints;
}

OrthodonticMetrics BiteSimulator::computeMetrics() const {
    OrthodonticMetrics metrics;
    if (!m_mandible || !m_maxilla) return metrics;

    // Basic position metrics
    float tx = m_currentTranslation.x();
    float ty = m_currentTranslation.y();
    float tz = m_currentTranslation.z();
    metrics.overjet_mm = 2.0 + ty;
    metrics.overbite_mm = tz;
    metrics.midline_dev_mm = std::abs(tx);

    // Get signed distances and vertex data
    auto signedDists = computeSignedDistances();
    if (signedDists.empty()) return metrics;

    const auto& mandibleVerts = m_mandible->vertices();

    // Compute mandible centroid for determining anterior/posterior
    Eigen::Vector3f mandibleCentroid = Eigen::Vector3f::Zero();
    for (const auto& v : mandibleVerts) {
        mandibleCentroid += v.position;
    }
    mandibleCentroid /= static_cast<float>(mandibleVerts.size());

    // Contact thresholds
    const double contactThreshold = 0.3;    // Good contact: 0 ~ 0.3mm
    const double penetrationThreshold = -0.1;  // Penetration: < -0.1mm

    // Accumulators for 4 principles
    double forceAnterior = 0.0, forcePosterior = 0.0;
    double forceLeft = 0.0, forceRight = 0.0;
    double axialAlignmentSum = 0.0;
    double lateralForceSum = 0.0;
    int contactCount = 0;
    int penetrationCount = 0;

    // Regional force accumulators (for evenness calculation)
    // Divide into 4 quadrants: UL, UR, LL, LR (based on mesh position)
    double forceQuadrant[4] = {0, 0, 0, 0};  // LL, LR, UL, UR

    for (size_t i = 0; i < signedDists.size(); ++i) {
        double dist = signedDists[i];
        const auto& pos = mandibleVerts[i].position;
        const auto& normal = mandibleVerts[i].normal;

        // Skip if too far
        if (dist > contactThreshold) continue;

        // Compute "force" based on penetration depth
        // Force is proportional to how much the teeth are pressing
        // Positive dist = gap, negative dist = penetration
        double force = 0.0;
        if (dist <= 0) {
            // Penetration - force proportional to penetration depth
            force = std::min(1.0, std::abs(dist) / 0.5);  // Max force at 0.5mm penetration
            penetrationCount++;
        } else if (dist < contactThreshold) {
            // Light contact - small force
            force = 0.2 * (1.0 - dist / contactThreshold);
        }

        if (force < 0.01) continue;  // Skip negligible contacts
        contactCount++;

        // === 제1원칙: Anterior vs Posterior classification ===
        // Use Y coordinate relative to centroid (positive Y = anterior/front)
        // This is a simplification - ideally we'd use tooth segmentation
        bool isAnterior = (pos.y() > mandibleCentroid.y());

        if (isAnterior) {
            forceAnterior += force;
        } else {
            forcePosterior += force;
        }

        // === 제2원칙: Left vs Right ===
        bool isLeft = (pos.x() < mandibleCentroid.x());
        if (isLeft) {
            forceLeft += force;
        } else {
            forceRight += force;
        }

        // Quadrant (for evenness)
        int quadrant = (isLeft ? 0 : 1) + (isAnterior ? 2 : 0);
        forceQuadrant[quadrant] += force;

        // === 제3원칙: Axial Loading ===
        // Check if force direction is vertical (aligned with tooth axis)
        // Assume tooth axis is approximately vertical (Z-axis in most dental coordinate systems)
        // Contact normal points from mandible to maxilla
        Eigen::Vector3f toothAxis(0, 0, 1);  // Idealized vertical axis

        // Get the contact direction (from mandible to maxilla)
        int nearestMaxillaIdx = m_spatialHash.findNearest(pos);
        Eigen::Vector3f maxillaNormal = (nearestMaxillaIdx >= 0) ?
            m_maxillaNormals[nearestMaxillaIdx] : Eigen::Vector3f::UnitZ();

        // Force direction should be along tooth axis
        // Dot product of contact normal with vertical axis
        double axialAlignment = std::abs(maxillaNormal.dot(toothAxis));

        // Weight by force magnitude
        axialAlignmentSum += axialAlignment * force;

        // Lateral force component (force NOT along tooth axis)
        double lateralComponent = std::sqrt(1.0 - axialAlignment * axialAlignment);
        lateralForceSum += lateralComponent * force;
    }

    // === Calculate final metrics ===

    double totalForce = forceAnterior + forcePosterior + 1e-6;

    // 제1원칙: Protection ratio (구치부 힘 비율)
    metrics.force_anterior = forceAnterior;
    metrics.force_posterior = forcePosterior;
    metrics.protection_ratio = forcePosterior / totalForce;

    // 제2원칙: Balance
    metrics.force_left = forceLeft;
    metrics.force_right = forceRight;
    double totalLR = forceLeft + forceRight + 1e-6;
    metrics.balance_ratio = forceLeft / totalLR;
    metrics.balance_error = std::abs(forceLeft - forceRight) / totalLR;

    // 제3원칙: Axial loading
    if (contactCount > 0) {
        metrics.axial_alignment_score = axialAlignmentSum / (forceAnterior + forcePosterior + 1e-6);
        metrics.lateral_force_penalty = lateralForceSum / (forceAnterior + forcePosterior + 1e-6);
    }

    // 제4원칙: Contact evenness (variance of quadrant forces)
    metrics.contact_point_count = contactCount;
    metrics.penetration_count = penetrationCount;

    double meanQuadrantForce = (forceQuadrant[0] + forceQuadrant[1] + forceQuadrant[2] + forceQuadrant[3]) / 4.0;
    double variance = 0.0;
    for (int q = 0; q < 4; ++q) {
        variance += std::pow(forceQuadrant[q] - meanQuadrantForce, 2);
    }
    variance /= 4.0;
    // Evenness: inverse of coefficient of variation (lower variance = more even)
    metrics.contact_evenness = (meanQuadrantForce > 0.01) ? 1.0 / (1.0 + std::sqrt(variance) / meanQuadrantForce) : 0.0;

    // Legacy compatibility
    metrics.anterior_contact_ratio = metrics.force_anterior / (totalForce + 1e-6);
    metrics.posterior_contact_ratio = metrics.force_posterior / (totalForce + 1e-6);
    metrics.left_contact_force = forceLeft;
    metrics.right_contact_force = forceRight;

    return metrics;
}

double BiteSimulator::computeReward(const OrthodonticMetrics& m) const {
    // =====================================================
    // 교정과 전문의 기준 교합 점수 (4대 원칙 기반)
    // =====================================================

    double score = 0.0;

    // === 제1원칙: 상호 보호 교합 (Mutually Protected Occlusion) ===
    // 목표: 구치부가 힘의 90% 이상 받아야 함
    // protection_ratio = posterior / total, 이상적 값: 0.9+
    double protection_score = 0.0;
    if (m.protection_ratio >= 0.90) {
        protection_score = 1.0;  // 만점
    } else if (m.protection_ratio >= 0.70) {
        protection_score = 0.5 + 0.5 * (m.protection_ratio - 0.70) / 0.20;
    } else {
        protection_score = m.protection_ratio / 0.70 * 0.5;
    }
    // 전치부에 힘이 많이 걸리면 큰 감점
    double anterior_penalty = std::max(0.0, m.force_anterior - 0.1 * (m.force_anterior + m.force_posterior));
    protection_score -= anterior_penalty * 2.0;
    protection_score = std::max(0.0, protection_score);

    // === 제2원칙: 좌우 균형 (Bilateral Balance) ===
    // 목표: balance_error가 0에 가까울수록 좋음 (좌우 50:50)
    double balance_score = std::exp(-m.balance_error * m.balance_error * 25.0);  // e^(-error^2 * 25)

    // === 제3원칙: 치축 방향 부하 (Axial Loading) ===
    // 목표: axial_alignment_score가 1.0에 가까울수록 좋음
    double axial_score = m.axial_alignment_score;
    // 측방력이 크면 큰 페널티
    double lateral_penalty = m.lateral_force_penalty * 2.0;
    axial_score = std::max(0.0, axial_score - lateral_penalty);

    // === 제4원칙: 접촉 분포 균일도 ===
    double evenness_score = m.contact_evenness;

    // === 페널티 ===
    // 침투(과교합) 페널티
    double penetration_penalty = std::min(1.0, m.penetration_count / 100.0);

    // 접촉점이 너무 적으면 페널티
    double contact_bonus = std::min(1.0, m.contact_point_count / 500.0);

    // === 가중치 적용 최종 점수 ===
    // 가중치: 교정과 전문의 기준 중요도 반영
    const double W_PROTECTION = 2.5;    // 가장 중요: 앞니 보호
    const double W_BALANCE = 2.0;       // 매우 중요: 좌우 균형
    const double W_AXIAL = 1.5;         // 중요: 수직 하중
    const double W_EVENNESS = 1.0;      // 보조: 균일 분포
    const double W_PENETRATION = -1.5;  // 페널티
    const double W_CONTACT = 0.5;       // 보너스: 접촉점 존재

    score = W_PROTECTION * protection_score +
            W_BALANCE * balance_score +
            W_AXIAL * axial_score +
            W_EVENNESS * evenness_score +
            W_PENETRATION * penetration_penalty +
            W_CONTACT * contact_bonus;

    // Normalize to -1 ~ 1 range
    // Max possible: 2.5 + 2.0 + 1.5 + 1.0 + 0.5 = 7.5
    // Min possible: -1.5
    double normalized = (score + 1.5) / 9.0 * 2.0 - 1.0;

    return std::clamp(normalized, -1.0, 1.0);
}

void BiteSimulator::runICPAlignment(int iterations) {
    if (!m_mandible || !m_maxilla) return;

    std::cout << "Running ICP alignment (" << iterations << " iterations)..." << std::endl;

    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<Eigen::Vector3f> mandibleVerts;
        mandibleVerts.reserve(m_mandible->vertexCount());
        for (const auto& v : m_mandible->vertices()) {
            mandibleVerts.push_back(v.position);
        }

        std::vector<Eigen::Vector3f> closestPoints;
        std::vector<double> distances;
        findClosestPoints(mandibleVerts, closestPoints, distances);

        Eigen::Vector3f srcCentroid = Eigen::Vector3f::Zero();
        Eigen::Vector3f dstCentroid = Eigen::Vector3f::Zero();

        for (size_t i = 0; i < mandibleVerts.size(); ++i) {
            srcCentroid += mandibleVerts[i];
            dstCentroid += closestPoints[i];
        }
        srcCentroid /= static_cast<float>(mandibleVerts.size());
        dstCentroid /= static_cast<float>(closestPoints.size());

        Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
        for (size_t i = 0; i < mandibleVerts.size(); ++i) {
            Eigen::Vector3f srcP = mandibleVerts[i] - srcCentroid;
            Eigen::Vector3f dstP = closestPoints[i] - dstCentroid;
            H += srcP * dstP.transpose();
        }

        Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3f R = svd.matrixV() * svd.matrixU().transpose();

        if (R.determinant() < 0) {
            Eigen::Matrix3f V = svd.matrixV();
            V.col(2) *= -1;
            R = V * svd.matrixU().transpose();
        }

        Eigen::Vector3f t = dstCentroid - R * srcCentroid;

        auto& verts = const_cast<std::vector<Vertex>&>(m_mandible->vertices());
        for (auto& v : verts) {
            v.position = R * v.position + t;
            v.normal = R * v.normal;
        }

        m_currentTranslation += t;

        Eigen::Matrix4f deltaMat = Eigen::Matrix4f::Identity();
        deltaMat.block<3,3>(0,0) = R;
        deltaMat.block<3,1>(0,3) = t;
        m_transformMatrix = deltaMat * m_transformMatrix;

        double meanDist = 0;
        for (double d : distances) meanDist += d;
        meanDist /= distances.size();

        if (iter % 5 == 0) {
            std::cout << "  Iteration " << iter << ": mean distance = " << meanDist << std::endl;
        }

        if (meanDist < 0.01) {
            std::cout << "  Converged at iteration " << iter << std::endl;
            break;
        }
    }
}

void BiteSimulator::optimizeStep(double learningRate) {
    if (!m_mandible || !m_maxilla) return;

    OrthodonticMetrics metrics = computeMetrics();
    double currentReward = computeReward(metrics);

    const float eps = 0.1f;

    Eigen::Vector3f bestDeltaT = Eigen::Vector3f::Zero();
    Eigen::Vector3f bestDeltaR = Eigen::Vector3f::Zero();
    double bestReward = currentReward;

    for (int axis = 0; axis < 3; ++axis) {
        for (float sign : {-1.0f, 1.0f}) {
            Eigen::Vector3f deltaT = Eigen::Vector3f::Zero();
            deltaT[axis] = sign * eps;

            applyTransform(Eigen::Vector3f::Zero(), deltaT);
            OrthodonticMetrics newMetrics = computeMetrics();
            double newReward = computeReward(newMetrics);

            if (newReward > bestReward) {
                bestReward = newReward;
                bestDeltaT = deltaT * learningRate;
                bestDeltaR = Eigen::Vector3f::Zero();
            }

            applyTransform(Eigen::Vector3f::Zero(), -deltaT);
        }
    }

    if (bestReward > currentReward) {
        applyTransform(bestDeltaR, bestDeltaT);
    }
}

bool BiteSimulator::saveMandible(const std::string& path) const {
    if (!m_mandible) return false;
    return MeshLoader::save(*m_mandible, path);
}
