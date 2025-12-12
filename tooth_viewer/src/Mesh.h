#ifndef MESH_H
#define MESH_H

#include <vector>
#include <string>
#include <Eigen/Core>
#include <Eigen/Geometry>

struct Vertex {
    Eigen::Vector3f position;
    Eigen::Vector3f normal;
    Eigen::Vector3f color;
};

struct Face {
    unsigned int v0, v1, v2;
};

class Mesh {
public:
    Mesh();
    ~Mesh();

    void clear();
    bool isEmpty() const;

    void addVertex(const Eigen::Vector3f& pos, const Eigen::Vector3f& normal = Eigen::Vector3f::Zero(),
                   const Eigen::Vector3f& color = Eigen::Vector3f(0.8f, 0.8f, 0.8f));
    void addFace(unsigned int v0, unsigned int v1, unsigned int v2);

    void computeNormals();
    void computeBoundingBox();
    void centerAndNormalize();
    void centerAndNormalizeWith(const Eigen::Vector3f& center, float scale);

    // Get original transformation parameters (after centerAndNormalize)
    Eigen::Vector3f originalCenter() const { return m_originalCenter; }
    float originalScale() const { return m_originalScale; }

    // Getters
    const std::vector<Vertex>& vertices() const { return m_vertices; }
    const std::vector<Face>& faces() const { return m_faces; }
    const std::vector<int>& labels() const { return m_labels; }
    size_t vertexCount() const { return m_vertices.size(); }
    size_t faceCount() const { return m_faces.size(); }
    bool isPointCloud() const { return m_faces.empty(); }
    bool hasLabels() const { return !m_labels.empty(); }

    // Labels
    void setLabel(size_t vertexIndex, int label);
    void reserveLabels(size_t count);

    Eigen::Vector3f center() const { return m_center; }
    float boundingRadius() const { return m_boundingRadius; }

    // For OpenGL
    std::vector<float> getVertexBuffer() const;
    std::vector<unsigned int> getIndexBuffer() const;

    // Color operations
    void setUniformColor(const Eigen::Vector3f& color);
    void setFaceColors(const std::vector<Eigen::Vector3f>& colors);

private:
    std::vector<Vertex> m_vertices;
    std::vector<Face> m_faces;
    std::vector<int> m_labels;  // Per-vertex labels (for segmentation)

    Eigen::Vector3f m_center;
    Eigen::Vector3f m_minBound;
    Eigen::Vector3f m_maxBound;
    float m_boundingRadius;

    // Original transformation parameters (for aligning point clouds)
    Eigen::Vector3f m_originalCenter = Eigen::Vector3f::Zero();
    float m_originalScale = 1.0f;
};

#endif // MESH_H
