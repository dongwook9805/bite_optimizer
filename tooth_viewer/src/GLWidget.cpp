#include "GLWidget.h"
#include <QMouseEvent>
#include <QWheelEvent>
#include <QKeyEvent>
#include <cmath>
#include <iostream>

GLWidget::GLWidget(QWidget* parent)
    : QOpenGLWidget(parent)
    , m_meshVbo(QOpenGLBuffer::VertexBuffer)
    , m_meshEbo(QOpenGLBuffer::IndexBuffer)
    , m_pcVbo(QOpenGLBuffer::VertexBuffer)
    , m_maxillaVbo(QOpenGLBuffer::VertexBuffer)
    , m_maxillaEbo(QOpenGLBuffer::IndexBuffer)
    , m_mandibleVbo(QOpenGLBuffer::VertexBuffer)
    , m_mandibleEbo(QOpenGLBuffer::IndexBuffer)
    , m_contactVbo(QOpenGLBuffer::VertexBuffer)
    , m_landmarkVbo(QOpenGLBuffer::VertexBuffer)
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
    m_maxillaVao.destroy();
    m_maxillaVbo.destroy();
    m_maxillaEbo.destroy();
    m_mandibleVao.destroy();
    m_mandibleVbo.destroy();
    m_mandibleEbo.destroy();
    m_contactVao.destroy();
    m_contactVbo.destroy();
    m_landmarkVao.destroy();
    m_landmarkVbo.destroy();
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

    // Create maxilla VAO/VBO/EBO
    m_maxillaVao.create();
    m_maxillaVbo.create();
    m_maxillaEbo.create();

    // Create mandible VAO/VBO/EBO
    m_mandibleVao.create();
    m_mandibleVbo.create();
    m_mandibleEbo.create();

    // Create contact points VAO/VBO
    m_contactVao.create();
    m_contactVbo.create();

    // Create landmark VAO/VBO
    m_landmarkVao.create();
    m_landmarkVbo.create();
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
    m_projection.perspective(45.0f, float(w) / float(h), 0.1f, 10000.0f);
}

void GLWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    m_shaderProgram->bind();

    // Update view matrix - orbit camera around target
    m_view.setToIdentity();
    m_view.translate(0, 0, -m_zoom);  // Move camera back
    m_view.rotate(m_rotation);         // Rotate around target
    m_view.translate(-m_cameraTarget - m_panOffset);  // Move to look at target
    m_model.setToIdentity();

    // Dynamic light position based on mesh scale
    QVector3D lightPos = m_cameraTarget + QVector3D(m_meshScale * 2, m_meshScale * 2, m_meshScale * 2);
    QVector3D viewPos = m_cameraTarget + QVector3D(0, 0, m_zoom);

    m_shaderProgram->setUniformValue("model", m_model);
    m_shaderProgram->setUniformValue("view", m_view);
    m_shaderProgram->setUniformValue("projection", m_projection);
    m_shaderProgram->setUniformValue("lightPos", lightPos);
    m_shaderProgram->setUniformValue("viewPos", viewPos);

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

    // Draw maxilla (upper jaw)
    if (m_maxillaVisible && m_maxilla && !m_maxilla->isEmpty() && m_maxillaIndexCount > 0) {
        m_shaderProgram->setUniformValue("isPointCloud", false);
        m_maxillaVao.bind();

        if (m_wireframe) {
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        } else {
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }

        glDrawElements(GL_TRIANGLES, m_maxillaIndexCount, GL_UNSIGNED_INT, nullptr);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        m_maxillaVao.release();
    }

    // Draw mandible (lower jaw)
    if (m_mandibleVisible && m_mandible && !m_mandible->isEmpty() && m_mandibleIndexCount > 0) {
        m_shaderProgram->setUniformValue("isPointCloud", false);
        m_mandibleVao.bind();

        if (m_wireframe) {
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        } else {
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }

        glDrawElements(GL_TRIANGLES, m_mandibleIndexCount, GL_UNSIGNED_INT, nullptr);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        m_mandibleVao.release();
    }

    // Draw contact points (green = good, red = penetration)
    if (m_contactPointsVisible && m_contactPointCount > 0) {
        m_shaderProgram->setUniformValue("isPointCloud", true);
        m_shaderProgram->setUniformValue("pointSize", 8.0f);

        glDisable(GL_CULL_FACE);
        glDisable(GL_DEPTH_TEST);  // Draw on top of meshes
        m_contactVao.bind();
        glDrawArrays(GL_POINTS, 0, m_contactPointCount);
        m_contactVao.release();
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
    }

    // Draw landmarks (large colored spheres)
    if (!m_landmarks.empty()) {
        m_shaderProgram->setUniformValue("isPointCloud", true);
        m_shaderProgram->setUniformValue("pointSize", 15.0f);  // Large points

        glDisable(GL_CULL_FACE);
        glDisable(GL_DEPTH_TEST);  // Draw on top
        m_landmarkVao.bind();
        glDrawArrays(GL_POINTS, 0, static_cast<int>(m_landmarks.size()));
        m_landmarkVao.release();
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

void GLWidget::resetView()
{
    // Reset to fit current meshes if available
    if ((m_maxilla && !m_maxilla->isEmpty()) || (m_mandible && !m_mandible->isEmpty())) {
        adjustCameraToFitMeshes();
    } else if (m_mesh && !m_mesh->isEmpty()) {
        m_zoom = 3.0f;
        m_rotation = QQuaternion();
        m_panOffset = QVector3D(0, 0, 0);
        m_cameraTarget = QVector3D(0, 0, 0);
        m_meshScale = 1.0f;
    } else {
        m_zoom = 3.0f;
        m_rotation = QQuaternion();
        m_panOffset = QVector3D(0, 0, 0);
        m_cameraTarget = QVector3D(0, 0, 0);
        m_meshScale = 1.0f;
    }
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

    // Landmark picking mode
    if (m_landmarkPickingMode && event->button() == Qt::LeftButton) {
        Eigen::Vector3f hitPoint;
        bool hitMaxilla;

        if (pickVertex(event->pos().x(), event->pos().y(), hitPoint, hitMaxilla)) {
            // Check if we picked the expected mesh
            if (hitMaxilla == m_expectingMaxilla) {
                Landmark lm;
                lm.position = hitPoint;
                lm.isMaxilla = hitMaxilla;
                lm.index = m_currentLandmarkPair;

                m_landmarks.push_back(lm);
                updateLandmarkBuffers();
                emit landmarkPicked(lm);

                // Toggle expectation
                if (m_expectingMaxilla) {
                    // Just picked maxilla, now expect mandible
                    m_expectingMaxilla = false;
                } else {
                    // Just picked mandible, pair complete
                    emit landmarkPairComplete(m_currentLandmarkPair);
                    m_currentLandmarkPair++;
                    m_expectingMaxilla = true;  // Next pair starts with maxilla
                }
                update();
            } else {
                std::cout << "Wrong mesh! Expected " << (m_expectingMaxilla ? "maxilla (upper)" : "mandible (lower)") << std::endl;
            }
        }
        return;  // Don't rotate when in picking mode
    }

    // Check if clicking on a mesh to select and drag it (Left-click on mesh)
    if (m_biteSimulator && event->button() == Qt::LeftButton) {
        Eigen::Vector3f hitPoint;
        bool hitMaxilla;

        // Ctrl+Left-click = rotate mode
        bool ctrlHeld = (event->modifiers() & Qt::ControlModifier);

        if (pickVertex(event->pos().x(), event->pos().y(), hitPoint, hitMaxilla)) {
            // Clicked on a mesh - select it and start dragging
            m_movingMaxilla = hitMaxilla;
            m_draggingMesh = true;
            m_draggingRotate = ctrlHeld;
            setCursor(m_draggingRotate ? Qt::SizeAllCursor : Qt::OpenHandCursor);
            emit jawSelectionChanged(m_movingMaxilla);
            return;
        } else if (ctrlHeld && m_biteSimulator->maxilla() && m_biteSimulator->mandible()) {
            // Ctrl+click on empty space = rotate currently selected jaw
            m_draggingMesh = true;
            m_draggingRotate = true;
            setCursor(Qt::SizeAllCursor);
            return;
        }
    }

    // Middle-click = Set rotation pivot (Exocad style)
    if (event->button() == Qt::MiddleButton) {
        Eigen::Vector3f hitPoint;
        bool hitMaxilla;
        if (pickVertex(event->pos().x(), event->pos().y(), hitPoint, hitMaxilla)) {
            // Set clicked point as rotation center
            m_cameraTarget = QVector3D(hitPoint.x(), hitPoint.y(), hitPoint.z());
            std::cout << "Rotation pivot set to: (" << hitPoint.x() << ", " << hitPoint.y() << ", " << hitPoint.z() << ")" << std::endl;
        }
        return;
    }

    // Right-click = View rotation (Exocad style)
    if (event->button() == Qt::RightButton) {
        m_rotating = true;
        setCursor(Qt::ClosedHandCursor);
    }

    // Left+Right simultaneous = Panning (Exocad style)
    if ((event->buttons() & Qt::LeftButton) && (event->buttons() & Qt::RightButton)) {
        m_panning = true;
        m_rotating = false;  // Override rotation
        setCursor(Qt::SizeAllCursor);
    }
}

void GLWidget::mouseReleaseEvent(QMouseEvent* event)
{
    Q_UNUSED(event);

    // If we were dragging a mesh, emit full update signal
    if (m_draggingMesh) {
        emit jawMoved();  // Trigger expensive contact/metrics calculation
    }

    m_rotating = false;
    m_panning = false;
    m_draggingMesh = false;
    m_draggingRotate = false;
    setCursor(Qt::ArrowCursor);
}

void GLWidget::mouseMoveEvent(QMouseEvent* event)
{
    QPoint delta = event->pos() - m_lastMousePos;

    // Dragging a mesh to move it
    if (m_draggingMesh && m_biteSimulator) {
        if (m_draggingRotate) {
            // Exocad style: Rotate around view-relative axes
            float rotateSpeed = 0.5f;

            // Get view-space axes in world coordinates
            QQuaternion invRotation = m_rotation.conjugated();
            QVector3D viewRight = invRotation.rotatedVector(QVector3D(1, 0, 0));
            QVector3D viewUp = invRotation.rotatedVector(QVector3D(0, 1, 0));

            // Mouse horizontal → rotate around view's up axis
            // Mouse vertical → rotate around view's right axis
            float angleH = -delta.x() * rotateSpeed;  // Horizontal mouse → Y rotation
            float angleV = -delta.y() * rotateSpeed;  // Vertical mouse → X rotation

            // Build rotation matrix from view-relative axes
            Eigen::Vector3f axisH(viewUp.x(), viewUp.y(), viewUp.z());
            Eigen::Vector3f axisV(viewRight.x(), viewRight.y(), viewRight.z());

            Eigen::AngleAxisf rotH(angleH * M_PI / 180.0f, axisH.normalized());
            Eigen::AngleAxisf rotV(angleV * M_PI / 180.0f, axisV.normalized());
            Eigen::Matrix3f rotationMatrix = (rotH * rotV).toRotationMatrix();

            // Apply rotation via BiteSimulator (same pattern as translation)
            m_biteSimulator->applyRotationMatrix(rotationMatrix, m_movingMaxilla);
        } else {
            // Translate the selected mesh in screen plane (view-relative)
            float moveSpeed = m_meshScale * 0.002f;

            // Get camera right and up vectors from rotation (use conjugate for view->world transform)
            QQuaternion invRotation = m_rotation.conjugated();
            QVector3D worldRight = invRotation.rotatedVector(QVector3D(1, 0, 0));
            QVector3D worldUp = invRotation.rotatedVector(QVector3D(0, 1, 0));

            // Compute world-space translation from screen delta
            // delta.x positive = mouse right, delta.y positive = mouse down
            QVector3D move = worldRight * (delta.x() * moveSpeed) + worldUp * (-delta.y() * moveSpeed);

            Eigen::Vector3f deltaTranslation(move.x(), move.y(), move.z());
            m_biteSimulator->applyTransform(Eigen::Vector3f::Zero(), deltaTranslation, m_movingMaxilla);
        }

        // DEBUG: Check if meshes are separate
        std::cout << "=== MESH POINTERS ===" << std::endl;
        std::cout << "  BiteSimulator maxilla: " << m_biteSimulator->maxilla() << std::endl;
        std::cout << "  BiteSimulator mandible: " << m_biteSimulator->mandible() << std::endl;
        std::cout << "  GLWidget maxilla: " << m_maxilla.get() << std::endl;
        std::cout << "  GLWidget mandible: " << m_mandible.get() << std::endl;
        std::cout << "  Moving: " << (m_movingMaxilla ? "MAXILLA" : "MANDIBLE") << std::endl;

        // Update visualization (fast - just mesh positions)
        if (m_movingMaxilla) {
            updateMaxillaFromSimulator(m_biteSimulator->maxilla());
        } else {
            updateMandibleFromSimulator(m_biteSimulator->mandible());
        }
        // Emit fast signal (skips expensive contact/metrics calculation)
        emit jawMovedFast();
        m_lastMousePos = event->pos();
        update();
        return;
    }

    // Exocad style: Left+Right = Pan (check first, takes priority)
    if ((event->buttons() & Qt::LeftButton) && (event->buttons() & Qt::RightButton)) {
        float panSpeed = m_zoom * 0.002f;
        m_panOffset += QVector3D(-delta.x() * panSpeed, delta.y() * panSpeed, 0);
    }
    // Exocad style: Right only = View rotation
    else if (m_rotating && (event->buttons() & Qt::RightButton)) {
        float rotationSpeed = 0.5f;
        float angleX = delta.y() * rotationSpeed;
        float angleY = delta.x() * rotationSpeed;

        QQuaternion rotX = QQuaternion::fromAxisAndAngle(QVector3D(1, 0, 0), angleX);
        QQuaternion rotY = QQuaternion::fromAxisAndAngle(QVector3D(0, 1, 0), angleY);

        m_rotation = rotY * rotX * m_rotation;
    }
    // Legacy panning (middle button)
    else if (m_panning) {
        float panSpeed = m_zoom * 0.002f;
        m_panOffset += QVector3D(-delta.x() * panSpeed, delta.y() * panSpeed, 0);
    }

    m_lastMousePos = event->pos();
    update();
}

void GLWidget::wheelEvent(QWheelEvent* event)
{
    // Smooth zoom - zoom speed proportional to current zoom
    float delta = event->angleDelta().y() / 120.0f;
    float zoomFactor = 1.0f - delta * 0.15f;
    m_zoom *= zoomFactor;

    // Clamp zoom based on mesh scale
    float minZoom = m_meshScale * 0.1f;
    float maxZoom = m_meshScale * 20.0f;
    m_zoom = std::max(minZoom, std::min(m_zoom, maxZoom));

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

void GLWidget::adjustCameraToFitMeshes()
{
    // Calculate combined bounding box of all loaded bite meshes
    Eigen::Vector3f minBound(std::numeric_limits<float>::max(),
                              std::numeric_limits<float>::max(),
                              std::numeric_limits<float>::max());
    Eigen::Vector3f maxBound(std::numeric_limits<float>::lowest(),
                              std::numeric_limits<float>::lowest(),
                              std::numeric_limits<float>::lowest());

    bool hasMesh = false;

    if (m_maxilla && !m_maxilla->isEmpty()) {
        for (const auto& v : m_maxilla->vertices()) {
            minBound = minBound.cwiseMin(v.position);
            maxBound = maxBound.cwiseMax(v.position);
        }
        hasMesh = true;
    }

    if (m_mandible && !m_mandible->isEmpty()) {
        for (const auto& v : m_mandible->vertices()) {
            minBound = minBound.cwiseMin(v.position);
            maxBound = maxBound.cwiseMax(v.position);
        }
        hasMesh = true;
    }

    if (!hasMesh) return;

    // Calculate center and size
    Eigen::Vector3f center = (minBound + maxBound) * 0.5f;
    float size = (maxBound - minBound).norm();

    // Store mesh scale for zoom limits
    m_meshScale = size;

    // Set camera target to mesh center
    m_cameraTarget = QVector3D(center.x(), center.y(), center.z());

    // Reset pan offset
    m_panOffset = QVector3D(0, 0, 0);

    // Set zoom to fit the mesh nicely
    m_zoom = size * 1.8f;

    // Reset rotation to default view (looking at front of teeth)
    m_rotation = QQuaternion();

    std::cout << "Camera adjusted - Center: (" << center.x() << ", " << center.y() << ", " << center.z()
              << "), Size: " << size << ", Zoom: " << m_zoom << std::endl;
}

// Bite optimization methods

void GLWidget::loadMaxilla(std::unique_ptr<Mesh> maxilla)
{
    m_maxilla = std::move(maxilla);
    if (!m_maxilla) return;

    // Set color for maxilla (beige/cream)
    m_maxilla->setUniformColor(Eigen::Vector3f(0.96f, 0.87f, 0.70f));

    // Keep original coordinates - just compute bounding box for camera
    m_maxilla->computeBoundingBox();

    // Debug: Print bounding box info
    Eigen::Vector3f minB(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    Eigen::Vector3f maxB(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());
    for (const auto& v : m_maxilla->vertices()) {
        minB = minB.cwiseMin(v.position);
        maxB = maxB.cwiseMax(v.position);
    }
    std::cout << "=== MAXILLA (Upper Jaw) ===" << std::endl;
    std::cout << "  Min: (" << minB.x() << ", " << minB.y() << ", " << minB.z() << ")" << std::endl;
    std::cout << "  Max: (" << maxB.x() << ", " << maxB.y() << ", " << maxB.z() << ")" << std::endl;
    std::cout << "  Center: (" << (minB.x()+maxB.x())/2 << ", " << (minB.y()+maxB.y())/2 << ", " << (minB.z()+maxB.z())/2 << ")" << std::endl;

    updateMaxillaBuffers();

    // Adjust camera to fit the mesh
    adjustCameraToFitMeshes();

    if (m_mandible) {
        emit biteDataLoaded();
    }

    update();
}

void GLWidget::loadMandible(std::unique_ptr<Mesh> mandible)
{
    m_mandible = std::move(mandible);
    if (!m_mandible) return;

    // Set color for mandible (light blue)
    m_mandible->setUniformColor(Eigen::Vector3f(0.68f, 0.85f, 0.90f));

    // Keep original coordinates - just compute bounding box
    m_mandible->computeBoundingBox();

    // Debug: Print bounding box info
    Eigen::Vector3f minB(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    Eigen::Vector3f maxB(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());
    for (const auto& v : m_mandible->vertices()) {
        minB = minB.cwiseMin(v.position);
        maxB = maxB.cwiseMax(v.position);
    }
    std::cout << "=== MANDIBLE (Lower Jaw) ===" << std::endl;
    std::cout << "  Min: (" << minB.x() << ", " << minB.y() << ", " << minB.z() << ")" << std::endl;
    std::cout << "  Max: (" << maxB.x() << ", " << maxB.y() << ", " << maxB.z() << ")" << std::endl;
    std::cout << "  Center: (" << (minB.x()+maxB.x())/2 << ", " << (minB.y()+maxB.y())/2 << ", " << (minB.z()+maxB.z())/2 << ")" << std::endl;

    updateMandibleBuffers();

    // Adjust camera to fit both meshes
    adjustCameraToFitMeshes();

    if (m_maxilla) {
        emit biteDataLoaded();
    }

    update();
}

void GLWidget::updateMaxillaFromSimulator(Mesh* maxilla)
{
    if (!maxilla || !m_maxilla) return;

    // Copy vertex positions from simulator's maxilla
    auto& verts = const_cast<std::vector<Vertex>&>(m_maxilla->vertices());
    const auto& simVerts = maxilla->vertices();

    for (size_t i = 0; i < verts.size() && i < simVerts.size(); ++i) {
        verts[i].position = simVerts[i].position;
        verts[i].normal = simVerts[i].normal;
    }

    updateMaxillaBuffers();
    update();
}

void GLWidget::updateMandibleFromSimulator(Mesh* mandible)
{
    if (!mandible || !m_mandible) return;

    // Copy vertex positions from simulator's mandible
    auto& verts = const_cast<std::vector<Vertex>&>(m_mandible->vertices());
    const auto& simVerts = mandible->vertices();

    for (size_t i = 0; i < verts.size() && i < simVerts.size(); ++i) {
        verts[i].position = simVerts[i].position;
        verts[i].normal = simVerts[i].normal;
    }

    updateMandibleBuffers();
    update();
}

void GLWidget::updateMandibleColors(const std::vector<Eigen::Vector3f>& colors)
{
    if (!m_mandible || colors.empty()) return;

    m_mandible->setVertexColors(colors);
    updateMandibleBuffers();
    update();
}

void GLWidget::updateMaxillaBuffers()
{
    if (!m_maxilla) return;

    makeCurrent();
    m_maxillaVao.bind();

    std::vector<float> vertexData = m_maxilla->getVertexBuffer();
    m_maxillaVbo.bind();
    m_maxillaVbo.allocate(vertexData.data(), vertexData.size() * sizeof(float));

    std::vector<unsigned int> indexData = m_maxilla->getIndexBuffer();
    m_maxillaEbo.bind();
    m_maxillaEbo.allocate(indexData.data(), indexData.size() * sizeof(unsigned int));
    m_maxillaIndexCount = indexData.size();

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    m_maxillaVao.release();
    doneCurrent();
}

void GLWidget::updateMandibleBuffers()
{
    if (!m_mandible) return;

    makeCurrent();
    m_mandibleVao.bind();

    std::vector<float> vertexData = m_mandible->getVertexBuffer();
    m_mandibleVbo.bind();
    m_mandibleVbo.allocate(vertexData.data(), vertexData.size() * sizeof(float));

    std::vector<unsigned int> indexData = m_mandible->getIndexBuffer();
    m_mandibleEbo.bind();
    m_mandibleEbo.allocate(indexData.data(), indexData.size() * sizeof(unsigned int));
    m_mandibleIndexCount = indexData.size();

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    m_mandibleVao.release();
    doneCurrent();
}

void GLWidget::setMaxillaVisible(bool visible)
{
    m_maxillaVisible = visible;
    update();
}

void GLWidget::setMandibleVisible(bool visible)
{
    m_mandibleVisible = visible;
    update();
}

void GLWidget::updateContactPoints(const std::vector<ContactPoint>& contactPoints)
{
    m_contactPoints = contactPoints;
    updateContactPointBuffers();
    update();
}

void GLWidget::updateContactPointBuffers()
{
    if (m_contactPoints.empty()) {
        m_contactPointCount = 0;
        return;
    }

    makeCurrent();
    m_contactVao.bind();

    // Build vertex buffer: position (3) + normal (3) + color (3) per vertex
    std::vector<float> vertexData;
    vertexData.reserve(m_contactPoints.size() * 9);

    for (const auto& cp : m_contactPoints) {
        // Position
        vertexData.push_back(cp.position.x());
        vertexData.push_back(cp.position.y());
        vertexData.push_back(cp.position.z());

        // Normal (just point up for points)
        vertexData.push_back(0.0f);
        vertexData.push_back(0.0f);
        vertexData.push_back(1.0f);

        // Color based on contact type
        switch (cp.type) {
            case ContactPoint::GOOD_CONTACT:
                // Green
                vertexData.push_back(0.2f);
                vertexData.push_back(0.9f);
                vertexData.push_back(0.3f);
                break;
            case ContactPoint::PENETRATION:
                // Red
                vertexData.push_back(0.95f);
                vertexData.push_back(0.2f);
                vertexData.push_back(0.2f);
                break;
            case ContactPoint::GAP:
            default:
                // Yellow/Orange for near-contact
                vertexData.push_back(1.0f);
                vertexData.push_back(0.7f);
                vertexData.push_back(0.2f);
                break;
        }
    }

    m_contactVbo.bind();
    m_contactVbo.allocate(vertexData.data(), vertexData.size() * sizeof(float));

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    m_contactPointCount = m_contactPoints.size();

    m_contactVao.release();
    doneCurrent();
}

void GLWidget::setContactPointsVisible(bool visible)
{
    m_contactPointsVisible = visible;
    update();
}

// ========== Landmark Picking Methods ==========

void GLWidget::setLandmarkPickingMode(bool enable)
{
    m_landmarkPickingMode = enable;
    if (enable) {
        setCursor(Qt::CrossCursor);
        m_expectingMaxilla = true;
        m_currentLandmarkPair = 0;
        m_landmarks.clear();
        updateLandmarkBuffers();
    } else {
        setCursor(Qt::ArrowCursor);
    }
    update();
}

void GLWidget::clearLandmarks()
{
    m_landmarks.clear();
    m_currentLandmarkPair = 0;
    m_expectingMaxilla = true;
    updateLandmarkBuffers();
    update();
}

int GLWidget::landmarkPairCount() const
{
    // Count complete pairs (both maxilla and mandible)
    int maxillaCount = 0, mandibleCount = 0;
    for (const auto& lm : m_landmarks) {
        if (lm.isMaxilla) maxillaCount++;
        else mandibleCount++;
    }
    return std::min(maxillaCount, mandibleCount);
}

void GLWidget::updateLandmarkBuffers()
{
    if (m_landmarks.empty()) {
        return;
    }

    makeCurrent();
    m_landmarkVao.bind();

    // Build vertex buffer: position (3) + normal (3) + color (3) per landmark
    std::vector<float> vertexData;
    vertexData.reserve(m_landmarks.size() * 9);

    // Colors for different pairs
    const std::vector<Eigen::Vector3f> colors = {
        Eigen::Vector3f(1.0f, 0.3f, 0.3f),   // Red - pair 0
        Eigen::Vector3f(0.3f, 1.0f, 0.3f),   // Green - pair 1
        Eigen::Vector3f(0.3f, 0.3f, 1.0f),   // Blue - pair 2
        Eigen::Vector3f(1.0f, 1.0f, 0.3f),   // Yellow - pair 3
        Eigen::Vector3f(1.0f, 0.3f, 1.0f),   // Magenta - pair 4
    };

    for (const auto& lm : m_landmarks) {
        // Position
        vertexData.push_back(lm.position.x());
        vertexData.push_back(lm.position.y());
        vertexData.push_back(lm.position.z());

        // Normal (dummy)
        vertexData.push_back(0.0f);
        vertexData.push_back(0.0f);
        vertexData.push_back(1.0f);

        // Color based on pair index
        Eigen::Vector3f color = colors[lm.index % colors.size()];
        // Make mandible landmarks slightly darker
        if (!lm.isMaxilla) {
            color *= 0.7f;
        }
        vertexData.push_back(color.x());
        vertexData.push_back(color.y());
        vertexData.push_back(color.z());
    }

    m_landmarkVbo.bind();
    m_landmarkVbo.allocate(vertexData.data(), vertexData.size() * sizeof(float));

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    m_landmarkVao.release();
    doneCurrent();
}

Eigen::Vector3f GLWidget::getCameraPosition()
{
    // Camera position in world space
    QMatrix4x4 invView = m_view.inverted();
    QVector4D camPos4 = invView * QVector4D(0, 0, 0, 1);
    return Eigen::Vector3f(camPos4.x(), camPos4.y(), camPos4.z());
}

Eigen::Vector3f GLWidget::screenToWorldRay(int x, int y)
{
    // Convert screen coordinates to normalized device coordinates
    float ndcX = (2.0f * x / width()) - 1.0f;
    float ndcY = 1.0f - (2.0f * y / height());

    // Create clip space position (at near plane)
    QVector4D clipPos(ndcX, ndcY, -1.0f, 1.0f);

    // Transform to view space
    QMatrix4x4 invProj = m_projection.inverted();
    QVector4D viewPos = invProj * clipPos;
    viewPos.setZ(-1.0f);
    viewPos.setW(0.0f);

    // Transform to world space
    QMatrix4x4 invView = m_view.inverted();
    QVector4D worldDir = invView * viewPos;

    Eigen::Vector3f dir(worldDir.x(), worldDir.y(), worldDir.z());
    return dir.normalized();
}

bool GLWidget::pickVertex(int mouseX, int mouseY, Eigen::Vector3f& hitPoint, bool& hitMaxilla)
{
    Eigen::Vector3f rayOrigin = getCameraPosition();
    Eigen::Vector3f rayDir = screenToWorldRay(mouseX, mouseY);

    float bestDist = std::numeric_limits<float>::max();
    bool found = false;

    // Helper to test intersection with a mesh
    auto testMesh = [&](Mesh* mesh, bool isMaxilla) {
        if (!mesh || !mesh->vertices().size()) return;

        for (const auto& v : mesh->vertices()) {
            // Ray-sphere test (sphere around each vertex)
            float sphereRadius = m_meshScale * 0.01f;  // Small sphere around vertex

            Eigen::Vector3f oc = rayOrigin - v.position;
            float a = rayDir.dot(rayDir);
            float b = 2.0f * oc.dot(rayDir);
            float c = oc.dot(oc) - sphereRadius * sphereRadius;
            float discriminant = b * b - 4 * a * c;

            if (discriminant > 0) {
                float t = (-b - std::sqrt(discriminant)) / (2.0f * a);
                if (t > 0 && t < bestDist) {
                    bestDist = t;
                    hitPoint = v.position;
                    hitMaxilla = isMaxilla;
                    found = true;
                }
            }
        }
    };

    // Test both meshes
    if (m_maxillaVisible && m_maxilla) {
        testMesh(m_maxilla.get(), true);
    }
    if (m_mandibleVisible && m_mandible) {
        testMesh(m_mandible.get(), false);
    }

    return found;
}

// ========== Manual Jaw Movement ==========

void GLWidget::setManualMoveMode(bool enable)
{
    m_manualMoveMode = enable;
    if (enable) {
        setFocus();  // Ensure we receive key events
    }
}

void GLWidget::keyPressEvent(QKeyEvent* event)
{
    // Always allow manual movement when bite data is loaded
    if (!m_biteSimulator || !m_biteSimulator->maxilla() || !m_biteSimulator->mandible()) {
        QOpenGLWidget::keyPressEvent(event);
        return;
    }

    // Movement amounts (in mm)
    const float translateStep = 0.3f;   // 0.3mm per key press
    const float rotateStep = 0.5f;      // 0.5 degree per key press

    Eigen::Vector3f deltaTranslation = Eigen::Vector3f::Zero();
    Eigen::Vector3f deltaRotation = Eigen::Vector3f::Zero();  // degrees

    bool moved = false;

    switch (event->key()) {
        // === Translation (Arrow keys + Page Up/Down) ===
        case Qt::Key_Left:
            deltaTranslation.x() = -translateStep;
            moved = true;
            break;
        case Qt::Key_Right:
            deltaTranslation.x() = translateStep;
            moved = true;
            break;
        case Qt::Key_Up:
            deltaTranslation.y() = translateStep;
            moved = true;
            break;
        case Qt::Key_Down:
            deltaTranslation.y() = -translateStep;
            moved = true;
            break;
        case Qt::Key_PageUp:
            deltaTranslation.z() = translateStep;
            moved = true;
            break;
        case Qt::Key_PageDown:
            deltaTranslation.z() = -translateStep;
            moved = true;
            break;

        // === Rotation (Q/E, A/D, W/S) ===
        case Qt::Key_Q:
            deltaRotation.x() = -rotateStep;  // Pitch down
            moved = true;
            break;
        case Qt::Key_E:
            deltaRotation.x() = rotateStep;   // Pitch up
            moved = true;
            break;
        case Qt::Key_A:
            deltaRotation.y() = -rotateStep;  // Yaw left
            moved = true;
            break;
        case Qt::Key_D:
            deltaRotation.y() = rotateStep;   // Yaw right
            moved = true;
            break;
        case Qt::Key_Z:
            deltaRotation.z() = -rotateStep;  // Roll CCW
            moved = true;
            break;
        case Qt::Key_C:
            deltaRotation.z() = rotateStep;   // Roll CW
            moved = true;
            break;

        // === Switch which jaw to move ===
        case Qt::Key_Tab:
            m_movingMaxilla = !m_movingMaxilla;
            std::cout << "Now moving: " << (m_movingMaxilla ? "MAXILLA (Upper)" : "MANDIBLE (Lower)") << std::endl;
            emit jawSelectionChanged(m_movingMaxilla);
            break;

        default:
            QOpenGLWidget::keyPressEvent(event);
            return;
    }

    if (moved) {
        // Shift key for fine control (smaller steps)
        if (event->modifiers() & Qt::ShiftModifier) {
            deltaTranslation *= 0.2f;
            deltaRotation *= 0.2f;
        }

        // Apply transform to the selected jaw
        m_biteSimulator->applyTransform(deltaRotation, deltaTranslation, m_movingMaxilla);

        // Update visualization for the moved mesh
        if (m_movingMaxilla) {
            updateMaxillaFromSimulator(m_biteSimulator->maxilla());
        } else {
            updateMandibleFromSimulator(m_biteSimulator->mandible());
        }

        // Emit signal for real-time metrics update
        emit jawMoved();

        update();
    }
}
