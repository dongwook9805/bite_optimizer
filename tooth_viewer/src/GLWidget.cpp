#include "GLWidget.h"
#include <QMouseEvent>
#include <QWheelEvent>
#include <cmath>
#include <iostream>

GLWidget::GLWidget(QWidget* parent)
    : QOpenGLWidget(parent)
    , m_meshVbo(QOpenGLBuffer::VertexBuffer)
    , m_meshEbo(QOpenGLBuffer::IndexBuffer)
    , m_pcVbo(QOpenGLBuffer::VertexBuffer)
    , m_lmVbo(QOpenGLBuffer::VertexBuffer)
{
    setFocusPolicy(Qt::StrongFocus);

    // Initialize all labels as visible (0-16)
    for (int i = 0; i <= 16; ++i) {
        m_visibleLabels.insert(i);
    }
}

GLWidget::~GLWidget()
{
    makeCurrent();
    m_meshVao.destroy();
    m_meshVbo.destroy();
    m_meshEbo.destroy();
    m_pcVao.destroy();
    m_pcVbo.destroy();
    m_lmVao.destroy();
    m_lmVbo.destroy();
    delete m_shaderProgram;
    doneCurrent();
}

void GLWidget::initializeGL()
{
    initializeOpenGLFunctions();

    glClearColor(0.15f, 0.15f, 0.18f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glEnable(GL_PROGRAM_POINT_SIZE);

    setupShaders();

    // Create mesh VAO/VBO/EBO
    m_meshVao.create();
    m_meshVbo.create();
    m_meshEbo.create();

    // Create point cloud VAO/VBO
    m_pcVao.create();
    m_pcVbo.create();

    // Create landmark VAO/VBO
    m_lmVao.create();
    m_lmVbo.create();
}

void GLWidget::setupShaders()
{
    m_shaderProgram = new QOpenGLShaderProgram(this);

    const char* vertexShaderSource = R"(
        #version 330 core
        layout(location = 0) in vec3 aPos;
        layout(location = 1) in vec3 aNormal;
        layout(location = 2) in vec3 aColor;

        out vec3 FragPos;
        out vec3 Normal;
        out vec3 Color;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform float pointSize;
        uniform bool isPointCloud;

        void main() {
            FragPos = vec3(model * vec4(aPos, 1.0));
            Normal = mat3(transpose(inverse(model))) * aNormal;
            Color = aColor;
            gl_Position = projection * view * vec4(FragPos, 1.0);
            if (isPointCloud) {
                gl_PointSize = pointSize;
            }
        }
    )";

    const char* fragmentShaderSource = R"(
        #version 330 core
        in vec3 FragPos;
        in vec3 Normal;
        in vec3 Color;

        out vec4 FragColor;

        uniform vec3 lightPos;
        uniform vec3 viewPos;
        uniform bool isPointCloud;

        void main() {
            if (isPointCloud) {
                FragColor = vec4(Color, 1.0);
            } else {
                float ambientStrength = 0.3;
                vec3 ambient = ambientStrength * vec3(1.0);

                vec3 norm = normalize(Normal);
                vec3 lightDir = normalize(lightPos - FragPos);
                float diff = max(dot(norm, lightDir), 0.0);
                vec3 diffuse = diff * vec3(1.0);

                float specularStrength = 0.3;
                vec3 viewDir = normalize(viewPos - FragPos);
                vec3 reflectDir = reflect(-lightDir, norm);
                float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
                vec3 specular = specularStrength * spec * vec3(1.0);

                vec3 result = (ambient + diffuse + specular) * Color;
                FragColor = vec4(result, 1.0);
            }
        }
    )";

    m_shaderProgram->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderSource);
    m_shaderProgram->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderSource);
    m_shaderProgram->link();
}

void GLWidget::resizeGL(int w, int h)
{
    m_projection.setToIdentity();
    m_projection.perspective(45.0f, float(w) / float(h), 0.01f, 100.0f);
}

void GLWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    m_shaderProgram->bind();

    // Update matrices
    m_view.setToIdentity();
    m_view.translate(m_panOffset);
    m_view.translate(0, 0, -m_zoom);
    m_view.rotate(m_rotation);
    m_model.setToIdentity();

    m_shaderProgram->setUniformValue("model", m_model);
    m_shaderProgram->setUniformValue("view", m_view);
    m_shaderProgram->setUniformValue("projection", m_projection);
    m_shaderProgram->setUniformValue("lightPos", QVector3D(5.0f, 5.0f, 5.0f));
    m_shaderProgram->setUniformValue("viewPos", QVector3D(0.0f, 0.0f, m_zoom));

    // Draw mesh
    if (m_meshVisible && m_mesh && !m_mesh->isEmpty() && m_meshIndexCount > 0) {
        m_shaderProgram->setUniformValue("isPointCloud", false);
        m_meshVao.bind();

        if (m_wireframe) {
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        } else {
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }

        glDrawElements(GL_TRIANGLES, m_meshIndexCount, GL_UNSIGNED_INT, nullptr);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        m_meshVao.release();
    }

    // Draw point cloud
    if (m_pointCloudVisible && m_pointCloud && m_filteredPcVertexCount > 0) {
        m_shaderProgram->setUniformValue("isPointCloud", true);
        m_shaderProgram->setUniformValue("pointSize", 5.0f);

        glDisable(GL_CULL_FACE);
        m_pcVao.bind();
        glDrawArrays(GL_POINTS, 0, m_filteredPcVertexCount);
        m_pcVao.release();
        glEnable(GL_CULL_FACE);
    }

    // Draw landmarks (larger points)
    if (m_landmarksVisible && m_landmarks && m_landmarkCount > 0) {
        m_shaderProgram->setUniformValue("isPointCloud", true);
        m_shaderProgram->setUniformValue("pointSize", 15.0f);  // Larger points for landmarks

        glDisable(GL_CULL_FACE);
        glDisable(GL_DEPTH_TEST);  // Draw landmarks on top
        m_lmVao.bind();
        glDrawArrays(GL_POINTS, 0, m_landmarkCount);
        m_lmVao.release();
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
    }

    m_shaderProgram->release();
}

void GLWidget::loadMesh(std::unique_ptr<Mesh> mesh)
{
    m_mesh = std::move(mesh);

    if (!m_mesh) return;

    m_mesh->centerAndNormalize();
    updateMeshBuffers();
    resetView();

    emit meshLoaded(m_mesh->vertexCount(), m_mesh->faceCount());
    update();
}

void GLWidget::loadPointCloud(std::unique_ptr<Mesh> pointCloud)
{
    m_pointCloud = std::move(pointCloud);

    if (!m_pointCloud) return;

    // Use labels from Mesh if available, otherwise reverse map from colors
    m_pointLabels.clear();
    const auto& vertices = m_pointCloud->vertices();

    if (m_pointCloud->hasLabels()) {
        // Use labels directly from the PLY file
        const auto& meshLabels = m_pointCloud->labels();
        m_pointLabels.assign(meshLabels.begin(), meshLabels.end());
        std::cout << "Using " << m_pointLabels.size() << " labels from PLY file" << std::endl;
    } else {
        // Fallback: reverse map from colors
        for (const auto& v : vertices) {
            int r = static_cast<int>(v.color.x() * 255);
            int g = static_cast<int>(v.color.y() * 255);
            int b = static_cast<int>(v.color.z() * 255);

            // Check for gingiva (gray)
            if (r == 125 && g == 125 && b == 125) {
                m_pointLabels.push_back(0);
            } else {
                // Simplified hash-based approach
                int label = ((r + g + b) % 16) + 1;
                m_pointLabels.push_back(label);
            }
        }
        std::cout << "Reverse-mapped " << m_pointLabels.size() << " labels from colors" << std::endl;
    }

    // Use same normalization as mesh (if mesh is loaded)
    // This ensures point cloud aligns with the mesh
    if (m_mesh) {
        // Apply same transformation: use mesh's original center and scale
        m_pointCloud->centerAndNormalizeWith(m_mesh->originalCenter(), m_mesh->originalScale());
    } else {
        m_pointCloud->centerAndNormalize();
    }

    updatePointCloudBuffers();
    update();
}

void GLWidget::updateMeshBuffers()
{
    if (!m_mesh) return;

    makeCurrent();
    m_meshVao.bind();

    std::vector<float> vertexData = m_mesh->getVertexBuffer();
    m_meshVbo.bind();
    m_meshVbo.allocate(vertexData.data(), vertexData.size() * sizeof(float));
    m_meshVertexCount = m_mesh->vertexCount();

    std::vector<unsigned int> indexData = m_mesh->getIndexBuffer();
    m_meshEbo.bind();
    m_meshEbo.allocate(indexData.data(), indexData.size() * sizeof(unsigned int));
    m_meshIndexCount = indexData.size();

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    m_meshVao.release();
    doneCurrent();
}

void GLWidget::updatePointCloudBuffers()
{
    if (!m_pointCloud) return;

    makeCurrent();
    m_pcVao.bind();

    rebuildFilteredPointCloud();

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    m_pcVao.release();
    doneCurrent();
}

void GLWidget::rebuildFilteredPointCloud()
{
    if (!m_pointCloud) return;

    const auto& vertices = m_pointCloud->vertices();
    std::vector<float> filteredData;

    for (size_t i = 0; i < vertices.size(); ++i) {
        int label = (i < m_pointLabels.size()) ? m_pointLabels[i] : 0;

        if (m_visibleLabels.count(label) > 0) {
            const auto& v = vertices[i];
            filteredData.push_back(v.position.x());
            filteredData.push_back(v.position.y());
            filteredData.push_back(v.position.z());
            filteredData.push_back(v.normal.x());
            filteredData.push_back(v.normal.y());
            filteredData.push_back(v.normal.z());
            filteredData.push_back(v.color.x());
            filteredData.push_back(v.color.y());
            filteredData.push_back(v.color.z());
        }
    }

    m_filteredPcVertexCount = filteredData.size() / 9;

    m_pcVbo.bind();
    m_pcVbo.allocate(filteredData.data(), filteredData.size() * sizeof(float));
}

void GLWidget::setMeshVisible(bool visible)
{
    m_meshVisible = visible;
    update();
}

void GLWidget::setPointCloudVisible(bool visible)
{
    m_pointCloudVisible = visible;
    update();
}

void GLWidget::setLabelVisible(int label, bool visible)
{
    if (visible) {
        m_visibleLabels.insert(label);
    } else {
        m_visibleLabels.erase(label);
    }

    makeCurrent();
    m_pcVao.bind();
    rebuildFilteredPointCloud();
    m_pcVao.release();
    doneCurrent();

    update();
}

void GLWidget::loadLandmarks(std::unique_ptr<Mesh> landmarks)
{
    m_landmarks = std::move(landmarks);

    if (!m_landmarks) return;

    // Use same normalization as mesh (if mesh is loaded)
    if (m_mesh) {
        m_landmarks->centerAndNormalizeWith(m_mesh->originalCenter(), m_mesh->originalScale());
    } else {
        m_landmarks->centerAndNormalize();
    }

    updateLandmarkBuffers();
    update();
}

void GLWidget::updateLandmarkBuffers()
{
    if (!m_landmarks) return;

    makeCurrent();
    m_lmVao.bind();

    std::vector<float> vertexData = m_landmarks->getVertexBuffer();
    m_lmVbo.bind();
    m_lmVbo.allocate(vertexData.data(), vertexData.size() * sizeof(float));
    m_landmarkCount = m_landmarks->vertexCount();

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    m_lmVao.release();
    doneCurrent();
}

void GLWidget::setLandmarksVisible(bool visible)
{
    m_landmarksVisible = visible;
    update();
}

void GLWidget::resetView()
{
    m_zoom = 3.0f;
    m_rotation = QQuaternion();
    m_panOffset = QVector3D(0, 0, 0);
    update();
}

void GLWidget::setWireframe(bool enable)
{
    m_wireframe = enable;
    update();
}

void GLWidget::mousePressEvent(QMouseEvent* event)
{
    m_lastMousePos = event->pos();

    if (event->button() == Qt::LeftButton) {
        m_rotating = true;
    } else if (event->button() == Qt::MiddleButton || event->button() == Qt::RightButton) {
        m_panning = true;
    }
}

void GLWidget::mouseMoveEvent(QMouseEvent* event)
{
    QPoint delta = event->pos() - m_lastMousePos;

    if (m_rotating) {
        QVector3D v1 = arcballVector(m_lastMousePos.x(), m_lastMousePos.y());
        QVector3D v2 = arcballVector(event->pos().x(), event->pos().y());

        float angle = std::acos(std::min(1.0f, QVector3D::dotProduct(v1, v2)));
        QVector3D axis = QVector3D::crossProduct(v1, v2).normalized();

        QQuaternion deltaRotation = QQuaternion::fromAxisAndAngle(axis, angle * 180.0f / M_PI * 2.0f);
        m_rotation = deltaRotation * m_rotation;
    }

    if (m_panning) {
        float panSpeed = 0.005f * m_zoom;
        m_panOffset += QVector3D(delta.x() * panSpeed, -delta.y() * panSpeed, 0);
    }

    m_lastMousePos = event->pos();
    update();
}

void GLWidget::wheelEvent(QWheelEvent* event)
{
    float delta = event->angleDelta().y() / 120.0f;
    m_zoom *= (1.0f - delta * 0.1f);
    m_zoom = std::max(0.1f, std::min(m_zoom, 100.0f));
    update();
}

QVector3D GLWidget::arcballVector(int x, int y)
{
    float fx = (2.0f * x / width() - 1.0f);
    float fy = (1.0f - 2.0f * y / height());

    float z2 = 1.0f - fx * fx - fy * fy;
    float z = z2 > 0 ? std::sqrt(z2) : 0;

    return QVector3D(fx, fy, z).normalized();
}
