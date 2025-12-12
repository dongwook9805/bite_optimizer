#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLShaderProgram>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <QMatrix4x4>
#include <QVector3D>
#include <memory>
#include <set>
#include "Mesh.h"

class GLWidget : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core
{
    Q_OBJECT

public:
    explicit GLWidget(QWidget* parent = nullptr);
    ~GLWidget();

    void loadMesh(std::unique_ptr<Mesh> mesh);
    void loadPointCloud(std::unique_ptr<Mesh> pointCloud);
    void loadLandmarks(std::unique_ptr<Mesh> landmarks);
    Mesh* mesh() const { return m_mesh.get(); }
    Mesh* pointCloud() const { return m_pointCloud.get(); }
    Mesh* landmarks() const { return m_landmarks.get(); }

    void resetView();
    void setWireframe(bool enable);

    // Visibility controls
    void setMeshVisible(bool visible);
    void setPointCloudVisible(bool visible);
    void setLandmarksVisible(bool visible);
    void setLabelVisible(int label, bool visible);
    bool isMeshVisible() const { return m_meshVisible; }
    bool isPointCloudVisible() const { return m_pointCloudVisible; }
    bool isLandmarksVisible() const { return m_landmarksVisible; }

signals:
    void meshLoaded(size_t vertices, size_t faces);

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;

private:
    void setupShaders();
    void updateMeshBuffers();
    void updatePointCloudBuffers();
    void updateLandmarkBuffers();
    void rebuildFilteredPointCloud();
    QVector3D arcballVector(int x, int y);

    // Mesh data (original mesh)
    std::unique_ptr<Mesh> m_mesh;
    // Point cloud data (segmentation result)
    std::unique_ptr<Mesh> m_pointCloud;
    std::vector<int> m_pointLabels;  // Label for each point
    // Landmarks data
    std::unique_ptr<Mesh> m_landmarks;

    // OpenGL objects for mesh
    QOpenGLShaderProgram* m_shaderProgram = nullptr;
    QOpenGLVertexArrayObject m_meshVao;
    QOpenGLBuffer m_meshVbo;
    QOpenGLBuffer m_meshEbo;

    // OpenGL objects for point cloud
    QOpenGLVertexArrayObject m_pcVao;
    QOpenGLBuffer m_pcVbo;

    // OpenGL objects for landmarks
    QOpenGLVertexArrayObject m_lmVao;
    QOpenGLBuffer m_lmVbo;

    // Matrices
    QMatrix4x4 m_projection;
    QMatrix4x4 m_view;
    QMatrix4x4 m_model;

    // Camera
    float m_zoom = 3.0f;
    QQuaternion m_rotation;

    // Mouse interaction
    QPoint m_lastMousePos;
    bool m_rotating = false;
    bool m_panning = false;
    QVector3D m_panOffset;

    // Render options
    bool m_wireframe = false;
    bool m_meshVisible = true;
    bool m_pointCloudVisible = true;
    bool m_landmarksVisible = true;
    std::set<int> m_visibleLabels;  // Which labels are visible

    // Counts for drawing
    int m_meshIndexCount = 0;
    int m_meshVertexCount = 0;
    int m_pcVertexCount = 0;
    int m_filteredPcVertexCount = 0;
    int m_landmarkCount = 0;
};

#endif // GLWIDGET_H
