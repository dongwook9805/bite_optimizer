#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QOpenGLWidget>
#include <QPainter>
#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLShaderProgram>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <QMatrix4x4>
#include <QVector3D>
#include <memory>
#include <set>
#include <map>
#include "Mesh.h"
#include "BiteSimulator.h"

// Tooth centroid for FDI label display
struct ToothCentroid {
    Eigen::Vector3f position;
    int segLabel;    // Original segmentation label (1-16)
    int fdiNumber;   // FDI notation (11-18, 21-28, 31-38, 41-48)
    bool isMaxilla;
};

// Landmark for alignment
struct Landmark {
    Eigen::Vector3f position;
    bool isMaxilla;  // true = maxilla, false = mandible
    int index;       // landmark pair index (0, 1, 2, ...)
};

class GLWidget : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core
{
    Q_OBJECT

public:
    explicit GLWidget(QWidget* parent = nullptr);
    ~GLWidget();

    void loadMesh(std::unique_ptr<Mesh> mesh);
    void loadPointCloud(std::unique_ptr<Mesh> pointCloud);
    void loadMaxillaSegmentation(std::unique_ptr<Mesh> segPoints);
    void loadMandibleSegmentation(std::unique_ptr<Mesh> segPoints);
    Mesh* mesh() const { return m_mesh.get(); }
    Mesh* pointCloud() const { return m_pointCloud.get(); }
    void setMaxillaSegVisible(bool visible);
    void setMandibleSegVisible(bool visible);
    bool isMaxillaSegVisible() const { return m_maxillaSegVisible; }
    bool isMandibleSegVisible() const { return m_mandibleSegVisible; }

    // FDI label display
    void setFDILabelsVisible(bool visible);  // For backward compatibility
    void setMaxillaFDILabelsVisible(bool visible);
    void setMandibleFDILabelsVisible(bool visible);
    bool isMaxillaFDILabelsVisible() const { return m_fdiLabelsMaxillaVisible; }
    bool isMandibleFDILabelsVisible() const { return m_fdiLabelsMandibleVisible; }
    void updateToothCentroids();

    // Bite optimization: dual mesh support
    void loadMaxilla(std::unique_ptr<Mesh> maxilla);
    void loadMandible(std::unique_ptr<Mesh> mandible);
    void updateMaxillaFromSimulator(Mesh* maxilla);
    void updateMandibleFromSimulator(Mesh* mandible);
    void updateMandibleColors(const std::vector<Eigen::Vector3f>& colors);
    Mesh* maxilla() const { return m_maxilla.get(); }
    Mesh* mandible() const { return m_mandible.get(); }
    void setMaxillaVisible(bool visible);
    void setMandibleVisible(bool visible);
    bool isMaxillaVisible() const { return m_maxillaVisible; }
    bool isMandibleVisible() const { return m_mandibleVisible; }

    // Contact point visualization
    void updateContactPoints(const std::vector<ContactPoint>& contactPoints);
    void setContactPointsVisible(bool visible);
    bool isContactPointsVisible() const { return m_contactPointsVisible; }

    // Landmark picking mode
    void setLandmarkPickingMode(bool enable);
    bool isLandmarkPickingMode() const { return m_landmarkPickingMode; }
    void clearLandmarks();
    const std::vector<Landmark>& landmarks() const { return m_landmarks; }
    int landmarkPairCount() const;  // Number of complete pairs

    // Manual jaw movement mode
    void setManualMoveMode(bool enable);
    bool isManualMoveMode() const { return m_manualMoveMode; }
    void setMovingMaxilla(bool moveMaxilla) { m_movingMaxilla = moveMaxilla; }
    bool isMovingMaxilla() const { return m_movingMaxilla; }

    // Set BiteSimulator reference for manual movement
    void setBiteSimulator(BiteSimulator* simulator) { m_biteSimulator = simulator; }

    void resetView();
    void setWireframe(bool enable);

    // Visibility controls
    void setMeshVisible(bool visible);
    void setPointCloudVisible(bool visible);
    void setLabelVisible(int label, bool visible);
    bool isMeshVisible() const { return m_meshVisible; }
    bool isPointCloudVisible() const { return m_pointCloudVisible; }

signals:
    void meshLoaded(size_t vertices, size_t faces);
    void biteDataLoaded();
    void landmarkPicked(const Landmark& landmark);
    void landmarkPairComplete(int pairIndex);
    void jawMoved();  // Emitted when user manually moves a jaw (expensive update)
    void jawMovedFast();  // Emitted during dragging (skip expensive calculations)
    void jawSelectionChanged(bool movingMaxilla);  // Emitted when Tab switches jaw

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

    void mousePressEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;
    void paintEvent(QPaintEvent* event) override;

private:
    void setupShaders();
    void updateMeshBuffers();
    void updatePointCloudBuffers();
    void updateMaxillaBuffers();
    void updateMandibleBuffers();
    void updateContactPointBuffers();
    void updateLandmarkBuffers();
    void rebuildFilteredPointCloud();
    void adjustCameraToFitMeshes();
    QVector3D arcballVector(int x, int y);

    // Ray casting for picking
    bool pickVertex(int mouseX, int mouseY, Eigen::Vector3f& hitPoint, bool& hitMaxilla);
    Eigen::Vector3f screenToWorldRay(int x, int y);
    Eigen::Vector3f getCameraPosition();

    // FDI label helpers
    QPoint projectToScreen(const Eigen::Vector3f& worldPos);
    int segLabelToFDI(int segLabel, bool isMaxilla);
    void calculateToothCentroidsFromMesh(Mesh* mesh, bool isMaxilla);

    // Mesh data (original mesh)
    std::unique_ptr<Mesh> m_mesh;
    // Point cloud data (segmentation result)
    std::unique_ptr<Mesh> m_pointCloud;
    std::vector<int> m_pointLabels;  // Label for each point

    // Bite optimization meshes
    std::unique_ptr<Mesh> m_maxilla;   // Upper jaw (fixed)
    std::unique_ptr<Mesh> m_mandible;  // Lower jaw (movable)

    // OpenGL objects for mesh
    QOpenGLShaderProgram* m_shaderProgram = nullptr;
    QOpenGLVertexArrayObject m_meshVao;
    QOpenGLBuffer m_meshVbo;
    QOpenGLBuffer m_meshEbo;

    // OpenGL objects for point cloud
    QOpenGLVertexArrayObject m_pcVao;
    QOpenGLBuffer m_pcVbo;

    // OpenGL objects for maxilla
    QOpenGLVertexArrayObject m_maxillaVao;
    QOpenGLBuffer m_maxillaVbo;
    QOpenGLBuffer m_maxillaEbo;

    // OpenGL objects for mandible
    QOpenGLVertexArrayObject m_mandibleVao;
    QOpenGLBuffer m_mandibleVbo;
    QOpenGLBuffer m_mandibleEbo;

    // Segmentation point clouds (for bite mode)
    std::unique_ptr<Mesh> m_maxillaSeg;
    std::unique_ptr<Mesh> m_mandibleSeg;
    QOpenGLVertexArrayObject m_maxillaSegVao;
    QOpenGLBuffer m_maxillaSegVbo;
    QOpenGLVertexArrayObject m_mandibleSegVao;
    QOpenGLBuffer m_mandibleSegVbo;
    int m_maxillaSegVertexCount = 0;
    int m_mandibleSegVertexCount = 0;
    bool m_maxillaSegVisible = true;
    bool m_mandibleSegVisible = true;

    // FDI label display
    std::vector<ToothCentroid> m_toothCentroids;
    bool m_fdiLabelsMaxillaVisible = true;
    bool m_fdiLabelsMandibleVisible = true;

    // OpenGL objects for contact points
    QOpenGLVertexArrayObject m_contactVao;
    QOpenGLBuffer m_contactVbo;
    std::vector<ContactPoint> m_contactPoints;

    // OpenGL objects for landmarks
    QOpenGLVertexArrayObject m_landmarkVao;
    QOpenGLBuffer m_landmarkVbo;
    std::vector<Landmark> m_landmarks;
    bool m_landmarkPickingMode = false;
    int m_currentLandmarkPair = 0;  // Which pair we're picking
    bool m_expectingMaxilla = true;  // Next click should be maxilla?

    // Manual jaw movement
    bool m_manualMoveMode = false;
    bool m_movingMaxilla = false;  // false = moving mandible (default)
    BiteSimulator* m_biteSimulator = nullptr;  // Reference for applying transforms

    // Matrices
    QMatrix4x4 m_projection;
    QMatrix4x4 m_view;
    QMatrix4x4 m_model;

    // Camera
    float m_zoom = 3.0f;
    QQuaternion m_rotation;
    QVector3D m_cameraTarget;  // Point camera looks at
    float m_meshScale = 1.0f;  // Scale factor for proper interaction

    // Mouse interaction
    QPoint m_lastMousePos;
    bool m_rotating = false;
    bool m_panning = false;
    QVector3D m_panOffset;

    // Mesh dragging (Ctrl+drag to move selected mesh)
    bool m_draggingMesh = false;
    bool m_draggingRotate = false;  // true = rotate, false = translate

    // Render options
    bool m_wireframe = false;
    bool m_meshVisible = true;
    bool m_pointCloudVisible = true;
    bool m_maxillaVisible = true;
    bool m_mandibleVisible = true;
    bool m_contactPointsVisible = true;
    std::set<int> m_visibleLabels;  // Which labels are visible

    // Counts for drawing
    int m_meshIndexCount = 0;
    int m_meshVertexCount = 0;
    int m_pcVertexCount = 0;
    int m_filteredPcVertexCount = 0;
    int m_maxillaIndexCount = 0;
    int m_mandibleIndexCount = 0;
    int m_contactPointCount = 0;
};

#endif // GLWIDGET_H
