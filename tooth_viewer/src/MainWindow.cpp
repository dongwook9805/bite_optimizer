#include "MainWindow.h"
#include "MeshLoader.h"
#include <QMenuBar>
#include <QToolBar>
#include <QStatusBar>
#include <QFileDialog>
#include <QMessageBox>
#include <QAction>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    setWindowTitle("ToothViewer");
    resize(1200, 800);

    setupUI();
    setupMenuBar();
    setupToolBar();
    setupStatusBar();
}

MainWindow::~MainWindow() = default;

void MainWindow::setupUI()
{
    m_glWidget = new GLWidget(this);
    setCentralWidget(m_glWidget);

    connect(m_glWidget, &GLWidget::meshLoaded, this, &MainWindow::onMeshLoaded);
}

void MainWindow::setupMenuBar()
{
    // File menu
    QMenu* fileMenu = menuBar()->addMenu(tr("&File"));

    QAction* openAction = fileMenu->addAction(tr("&Open..."));
    openAction->setShortcut(QKeySequence::Open);
    connect(openAction, &QAction::triggered, this, &MainWindow::openFile);

    QAction* saveAction = fileMenu->addAction(tr("&Save As..."));
    saveAction->setShortcut(QKeySequence::SaveAs);
    connect(saveAction, &QAction::triggered, this, &MainWindow::saveFile);

    fileMenu->addSeparator();

    QAction* exitAction = fileMenu->addAction(tr("E&xit"));
    exitAction->setShortcut(QKeySequence::Quit);
    connect(exitAction, &QAction::triggered, this, &QMainWindow::close);

    // View menu
    QMenu* viewMenu = menuBar()->addMenu(tr("&View"));

    QAction* wireframeAction = viewMenu->addAction(tr("&Wireframe"));
    wireframeAction->setCheckable(true);
    wireframeAction->setShortcut(Qt::Key_W);
    connect(wireframeAction, &QAction::toggled, this, &MainWindow::toggleWireframe);

    QAction* resetViewAction = viewMenu->addAction(tr("&Reset View"));
    resetViewAction->setShortcut(Qt::Key_R);
    connect(resetViewAction, &QAction::triggered, this, &MainWindow::resetView);

    // Help menu
    QMenu* helpMenu = menuBar()->addMenu(tr("&Help"));

    QAction* aboutAction = helpMenu->addAction(tr("&About"));
    connect(aboutAction, &QAction::triggered, this, [this]() {
        QMessageBox::about(this, tr("About ToothViewer"),
            tr("ToothViewer v1.0\n\n"
               "3D Mesh Viewer for Dental Applications\n\n"
               "Supports: OBJ, PLY, STL"));
    });
}

void MainWindow::setupToolBar()
{
    QToolBar* toolbar = addToolBar(tr("Main Toolbar"));
    toolbar->setMovable(false);

    QAction* openAction = toolbar->addAction(tr("Open"));
    connect(openAction, &QAction::triggered, this, &MainWindow::openFile);

    QAction* saveAction = toolbar->addAction(tr("Save"));
    connect(saveAction, &QAction::triggered, this, &MainWindow::saveFile);

    toolbar->addSeparator();

    QAction* resetAction = toolbar->addAction(tr("Reset View"));
    connect(resetAction, &QAction::triggered, this, &MainWindow::resetView);

    QAction* wireframeAction = toolbar->addAction(tr("Wireframe"));
    wireframeAction->setCheckable(true);
    connect(wireframeAction, &QAction::toggled, this, &MainWindow::toggleWireframe);
}

void MainWindow::setupStatusBar()
{
    m_statusLabel = new QLabel(tr("Ready"));
    statusBar()->addWidget(m_statusLabel);
}

void MainWindow::openFile()
{
    QString filter = tr("3D Mesh Files (*.obj *.ply *.stl);;OBJ Files (*.obj);;PLY Files (*.ply);;STL Files (*.stl);;All Files (*)");
    QString filePath = QFileDialog::getOpenFileName(this, tr("Open Mesh"), QString(), filter);

    if (filePath.isEmpty()) return;

    m_currentFilePath = filePath;

    auto mesh = MeshLoader::load(filePath.toStdString());
    if (mesh) {
        m_glWidget->loadMesh(std::move(mesh));
        setWindowTitle(QString("ToothViewer - %1").arg(QFileInfo(filePath).fileName()));
    } else {
        QMessageBox::warning(this, tr("Error"), tr("Failed to load mesh file."));
    }
}

void MainWindow::saveFile()
{
    if (!m_glWidget->mesh()) {
        QMessageBox::warning(this, tr("Error"), tr("No mesh loaded."));
        return;
    }

    QString filter = tr("OBJ Files (*.obj);;PLY Files (*.ply)");
    QString filePath = QFileDialog::getSaveFileName(this, tr("Save Mesh"), QString(), filter);

    if (filePath.isEmpty()) return;

    if (MeshLoader::save(*m_glWidget->mesh(), filePath.toStdString())) {
        m_statusLabel->setText(tr("Saved: %1").arg(QFileInfo(filePath).fileName()));
    } else {
        QMessageBox::warning(this, tr("Error"), tr("Failed to save mesh file."));
    }
}

void MainWindow::toggleWireframe(bool checked)
{
    m_glWidget->setWireframe(checked);
}

void MainWindow::resetView()
{
    m_glWidget->resetView();
}

void MainWindow::onMeshLoaded(size_t vertices, size_t faces)
{
    m_statusLabel->setText(QString("Vertices: %1 | Faces: %2")
                           .arg(vertices).arg(faces));
}
