#include "Mesh.h"
#include <limits>
#include <cmath>

Mesh::Mesh()
    : m_center(Eigen::Vector3f::Zero())
    , m_minBound(Eigen::Vector3f::Zero())
    , m_maxBound(Eigen::Vector3f::Zero())
    , m_boundingRadius(1.0f)
{
}

Mesh::~Mesh() = default;

void Mesh::clear()
{
    m_vertices.clear();
    m_faces.clear();
    m_labels.clear();
    m_center = Eigen::Vector3f::Zero();
    m_boundingRadius = 1.0f;
}

bool Mesh::isEmpty() const
{
    return m_vertices.empty() || m_faces.empty();
}

void Mesh::addVertex(const Eigen::Vector3f& pos, const Eigen::Vector3f& normal, const Eigen::Vector3f& color)
{
    Vertex v;
    v.position = pos;
    v.normal = normal;
    v.color = color;
    m_vertices.push_back(v);
}

void Mesh::addFace(unsigned int v0, unsigned int v1, unsigned int v2)
{
    Face f;
    f.v0 = v0;
    f.v1 = v1;
    f.v2 = v2;
    m_faces.push_back(f);
}

void Mesh::computeNormals()
{
    // Reset all normals to zero
    for (auto& v : m_vertices) {
        v.normal = Eigen::Vector3f::Zero();
    }

    // Accumulate face normals
    for (const auto& face : m_faces) {
        const Eigen::Vector3f& p0 = m_vertices[face.v0].position;
        const Eigen::Vector3f& p1 = m_vertices[face.v1].position;
        const Eigen::Vector3f& p2 = m_vertices[face.v2].position;

        Eigen::Vector3f edge1 = p1 - p0;
        Eigen::Vector3f edge2 = p2 - p0;
        Eigen::Vector3f faceNormal = edge1.cross(edge2);

        m_vertices[face.v0].normal += faceNormal;
        m_vertices[face.v1].normal += faceNormal;
        m_vertices[face.v2].normal += faceNormal;
    }

    // Normalize
    for (auto& v : m_vertices) {
        if (v.normal.norm() > 1e-6f) {
            v.normal.normalize();
        }
    }
}

void Mesh::computeBoundingBox()
{
    if (m_vertices.empty()) return;

    m_minBound = Eigen::Vector3f(std::numeric_limits<float>::max(),
                                  std::numeric_limits<float>::max(),
                                  std::numeric_limits<float>::max());
    m_maxBound = Eigen::Vector3f(std::numeric_limits<float>::lowest(),
                                  std::numeric_limits<float>::lowest(),
                                  std::numeric_limits<float>::lowest());

    for (const auto& v : m_vertices) {
        m_minBound = m_minBound.cwiseMin(v.position);
        m_maxBound = m_maxBound.cwiseMax(v.position);
    }

    m_center = (m_minBound + m_maxBound) * 0.5f;
    m_boundingRadius = (m_maxBound - m_minBound).norm() * 0.5f;
}

void Mesh::centerAndNormalize()
{
    computeBoundingBox();

    if (m_boundingRadius < 1e-6f) return;

    // Store original transformation parameters
    m_originalCenter = m_center;
    m_originalScale = m_boundingRadius;

    float scale = 1.0f / m_boundingRadius;
    for (auto& v : m_vertices) {
        v.position = (v.position - m_center) * scale;
    }

    m_center = Eigen::Vector3f::Zero();
    m_boundingRadius = 1.0f;
}

void Mesh::centerAndNormalizeWith(const Eigen::Vector3f& center, float scale)
{
    // Apply the same transformation as another mesh
    if (scale < 1e-6f) return;

    float invScale = 1.0f / scale;
    for (auto& v : m_vertices) {
        v.position = (v.position - center) * invScale;
    }

    m_originalCenter = center;
    m_originalScale = scale;
    m_center = Eigen::Vector3f::Zero();
    m_boundingRadius = 1.0f;
}

std::vector<float> Mesh::getVertexBuffer() const
{
    // Format: pos(3) + normal(3) + color(3) = 9 floats per vertex
    std::vector<float> buffer;
    buffer.reserve(m_vertices.size() * 9);

    for (const auto& v : m_vertices) {
        buffer.push_back(v.position.x());
        buffer.push_back(v.position.y());
        buffer.push_back(v.position.z());
        buffer.push_back(v.normal.x());
        buffer.push_back(v.normal.y());
        buffer.push_back(v.normal.z());
        buffer.push_back(v.color.x());
        buffer.push_back(v.color.y());
        buffer.push_back(v.color.z());
    }

    return buffer;
}

std::vector<unsigned int> Mesh::getIndexBuffer() const
{
    std::vector<unsigned int> buffer;
    buffer.reserve(m_faces.size() * 3);

    for (const auto& f : m_faces) {
        buffer.push_back(f.v0);
        buffer.push_back(f.v1);
        buffer.push_back(f.v2);
    }

    return buffer;
}

void Mesh::setUniformColor(const Eigen::Vector3f& color)
{
    for (auto& v : m_vertices) {
        v.color = color;
    }
}

void Mesh::setVertexColors(const std::vector<Eigen::Vector3f>& colors)
{
    if (colors.size() != m_vertices.size()) return;

    for (size_t i = 0; i < m_vertices.size(); ++i) {
        m_vertices[i].color = colors[i];
    }
}

void Mesh::setFaceColors(const std::vector<Eigen::Vector3f>& colors)
{
    if (colors.size() != m_faces.size()) return;

    // Reset vertex colors
    std::vector<Eigen::Vector3f> vertexColorSum(m_vertices.size(), Eigen::Vector3f::Zero());
    std::vector<int> vertexColorCount(m_vertices.size(), 0);

    for (size_t i = 0; i < m_faces.size(); ++i) {
        const Face& f = m_faces[i];
        vertexColorSum[f.v0] += colors[i];
        vertexColorSum[f.v1] += colors[i];
        vertexColorSum[f.v2] += colors[i];
        vertexColorCount[f.v0]++;
        vertexColorCount[f.v1]++;
        vertexColorCount[f.v2]++;
    }

    for (size_t i = 0; i < m_vertices.size(); ++i) {
        if (vertexColorCount[i] > 0) {
            m_vertices[i].color = vertexColorSum[i] / static_cast<float>(vertexColorCount[i]);
        }
    }
}

void Mesh::setLabel(size_t vertexIndex, int label)
{
    if (m_labels.size() <= vertexIndex) {
        m_labels.resize(vertexIndex + 1, 0);
    }
    m_labels[vertexIndex] = label;
}

void Mesh::reserveLabels(size_t count)
{
    m_labels.reserve(count);
}
