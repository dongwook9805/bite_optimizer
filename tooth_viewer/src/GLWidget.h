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
#include "Mesh.h"

class GLWidget : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core
{
    Q_OBJECT

public:
    explicit GLWidget(QWidget* parent = nullptr);
    ~GLWidget();

    void loadMesh(std::unique_ptr<Mesh> mesh);
    Mesh* mesh() const { return m_mesh.get(); }

    void resetView();
    void setWireframe(bool enable);

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
    QVector3D arcballVector(int x, int y);

    // Mesh data
    std::unique_ptr<Mesh> m_mesh;

    // OpenGL objects
    QOpenGLShaderProgram* m_shaderProgram = nullptr;
    QOpenGLVertexArrayObject m_vao;
    QOpenGLBuffer m_vbo;
    QOpenGLBuffer m_ebo;

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

    // Index count for drawing
    int m_indexCount = 0;
};

#endif // GLWIDGET_H
