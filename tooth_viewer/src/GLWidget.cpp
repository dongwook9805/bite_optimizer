#include "GLWidget.h"
#include <QMouseEvent>
#include <QWheelEvent>
#include <cmath>

GLWidget::GLWidget(QWidget* parent)
    : QOpenGLWidget(parent)
    , m_vbo(QOpenGLBuffer::VertexBuffer)
    , m_ebo(QOpenGLBuffer::IndexBuffer)
{
    setFocusPolicy(Qt::StrongFocus);
}

GLWidget::~GLWidget()
{
    makeCurrent();
    m_vao.destroy();
    m_vbo.destroy();
    m_ebo.destroy();
    delete m_shaderProgram;
    doneCurrent();
}

void GLWidget::initializeGL()
{
    initializeOpenGLFunctions();

    glClearColor(0.15f, 0.15f, 0.18f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    setupShaders();

    m_vao.create();
    m_vbo.create();
    m_ebo.create();
}

void GLWidget::setupShaders()
{
    m_shaderProgram = new QOpenGLShaderProgram(this);

    // Vertex shader
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

        void main() {
            FragPos = vec3(model * vec4(aPos, 1.0));
            Normal = mat3(transpose(inverse(model))) * aNormal;
            Color = aColor;
            gl_Position = projection * view * vec4(FragPos, 1.0);
        }
    )";

    // Fragment shader
    const char* fragmentShaderSource = R"(
        #version 330 core
        in vec3 FragPos;
        in vec3 Normal;
        in vec3 Color;

        out vec4 FragColor;

        uniform vec3 lightPos;
        uniform vec3 viewPos;

        void main() {
            // Ambient
            float ambientStrength = 0.3;
            vec3 ambient = ambientStrength * vec3(1.0);

            // Diffuse
            vec3 norm = normalize(Normal);
            vec3 lightDir = normalize(lightPos - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * vec3(1.0);

            // Specular
            float specularStrength = 0.3;
            vec3 viewDir = normalize(viewPos - FragPos);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
            vec3 specular = specularStrength * spec * vec3(1.0);

            vec3 result = (ambient + diffuse + specular) * Color;
            FragColor = vec4(result, 1.0);
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

    if (!m_mesh || m_mesh->isEmpty()) return;

    m_shaderProgram->bind();

    // Update view matrix
    m_view.setToIdentity();
    m_view.translate(m_panOffset);
    m_view.translate(0, 0, -m_zoom);
    m_view.rotate(m_rotation);

    // Update model matrix
    m_model.setToIdentity();

    // Set uniforms
    m_shaderProgram->setUniformValue("model", m_model);
    m_shaderProgram->setUniformValue("view", m_view);
    m_shaderProgram->setUniformValue("projection", m_projection);
    m_shaderProgram->setUniformValue("lightPos", QVector3D(5.0f, 5.0f, 5.0f));
    m_shaderProgram->setUniformValue("viewPos", QVector3D(0.0f, 0.0f, m_zoom));

    // Draw mesh
    m_vao.bind();

    if (m_wireframe) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    } else {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    glDrawElements(GL_TRIANGLES, m_indexCount, GL_UNSIGNED_INT, nullptr);

    m_vao.release();
    m_shaderProgram->release();
}

void GLWidget::loadMesh(std::unique_ptr<Mesh> mesh)
{
    m_mesh = std::move(mesh);

    if (!m_mesh) return;

    // Center and normalize
    m_mesh->centerAndNormalize();

    updateMeshBuffers();
    resetView();

    emit meshLoaded(m_mesh->vertexCount(), m_mesh->faceCount());
    update();
}

void GLWidget::updateMeshBuffers()
{
    if (!m_mesh) return;

    makeCurrent();

    m_vao.bind();

    // Vertex buffer
    std::vector<float> vertexData = m_mesh->getVertexBuffer();
    m_vbo.bind();
    m_vbo.allocate(vertexData.data(), vertexData.size() * sizeof(float));

    // Index buffer
    std::vector<unsigned int> indexData = m_mesh->getIndexBuffer();
    m_ebo.bind();
    m_ebo.allocate(indexData.data(), indexData.size() * sizeof(unsigned int));
    m_indexCount = indexData.size();

    // Vertex attributes
    // Position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
    // Normal
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // Color
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    m_vao.release();

    doneCurrent();
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
        // Arcball rotation
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
