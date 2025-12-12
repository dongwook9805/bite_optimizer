#include "MainWindow.h"
#include "MeshLoader.h"
#include <QMenuBar>
#include <QToolBar>
#include <QStatusBar>
#include <QFileDialog>
#include <QMessageBox>
#include <QAction>
#include <QTemporaryFile>
#include <QDir>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QPushButton>
#include <QScrollArea>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    setWindowTitle("ToothViewer");
    resize(1400, 900);

    m_segmentation = new Segmentation(this);
    connect(m_segmentation, &Segmentation::segmentationFinished,
            this, &MainWindow::onSegmentationFinished);
    connect(m_segmentation, &Segmentation::segmentationProgress,
            this, &MainWindow::onSegmentationProgress);
    connect(m_segmentation, &Segmentation::landmarksFinished,
            this, &MainWindow::onLandmarksFinished);

    setupUI();
    setupMenuBar();
    setupToolBar();
    setupStatusBar();
    setupSidePanel();
}

MainWindow::~MainWindow() = default;

void MainWindow::setupUI()
{
    m_glWidget = new GLWidget(this);
    setCentralWidget(m_glWidget);

    connect(m_glWidget, &GLWidget::meshLoaded, this, &MainWindow::onMeshLoaded);
}

void MainWindow::setupSidePanel()
{
    m_sidePanel = new QDockWidget(tr("Layers"), this);
    m_sidePanel->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable);
    m_sidePanel->setMinimumWidth(200);

    QWidget* panelWidget = new QWidget();
    QVBoxLayout* mainLayout = new QVBoxLayout(panelWidget);
    mainLayout->setContentsMargins(10, 10, 10, 10);
    mainLayout->setSpacing(10);

    // Mesh visibility group (initially hidden)
    m_meshGroup = new QGroupBox(tr("Mesh"));
    QVBoxLayout* meshLayout = new QVBoxLayout(m_meshGroup);

    m_meshVisibleCheck = new QCheckBox(tr("Show Mesh"));
    m_meshVisibleCheck->setChecked(true);
    connect(m_meshVisibleCheck, &QCheckBox::toggled, this, &MainWindow::onMeshVisibilityChanged);
    meshLayout->addWidget(m_meshVisibleCheck);

    mainLayout->addWidget(m_meshGroup);
    m_meshGroup->hide();  // Hidden until mesh is loaded

    // Point cloud / Segmentation group (initially hidden)
    m_segmentationGroup = new QGroupBox(tr("Segmentation"));
    QVBoxLayout* segLayout = new QVBoxLayout(m_segmentationGroup);

    m_pointCloudVisibleCheck = new QCheckBox(tr("Show All Points"));
    m_pointCloudVisibleCheck->setChecked(true);
    connect(m_pointCloudVisibleCheck, &QCheckBox::toggled, this, &MainWindow::onPointCloudVisibilityChanged);
    segLayout->addWidget(m_pointCloudVisibleCheck);

    // Select All/None buttons
    QHBoxLayout* buttonLayout = new QHBoxLayout();
    QPushButton* selectAllBtn = new QPushButton(tr("All"));
    QPushButton* selectNoneBtn = new QPushButton(tr("None"));
    buttonLayout->addWidget(selectAllBtn);
    buttonLayout->addWidget(selectNoneBtn);
    segLayout->addLayout(buttonLayout);

    connect(selectAllBtn, &QPushButton::clicked, this, [this]() {
        for (auto* cb : m_labelCheckboxes) {
            cb->setChecked(true);
        }
    });

    connect(selectNoneBtn, &QPushButton::clicked, this, [this]() {
        for (auto* cb : m_labelCheckboxes) {
            cb->setChecked(false);
        }
    });

    // Scroll area for labels
    QScrollArea* scrollArea = new QScrollArea();
    scrollArea->setWidgetResizable(true);
    scrollArea->setFrameShape(QFrame::NoFrame);

    m_labelsWidget = new QWidget();
    m_labelsLayout = new QVBoxLayout(m_labelsWidget);
    m_labelsLayout->setSpacing(2);

    m_labelsLayout->addStretch();
    scrollArea->setWidget(m_labelsWidget);
    segLayout->addWidget(scrollArea);

    mainLayout->addWidget(m_segmentationGroup, 1);
    m_segmentationGroup->hide();  // Hidden until segmentation is done

    // Landmarks group (initially hidden)
    m_landmarksGroup = new QGroupBox(tr("Landmarks"));
    QVBoxLayout* lmLayout = new QVBoxLayout(m_landmarksGroup);

    m_landmarksVisibleCheck = new QCheckBox(tr("Show Landmarks"));
    m_landmarksVisibleCheck->setChecked(true);
    connect(m_landmarksVisibleCheck, &QCheckBox::toggled, this, [this](bool visible) {
        m_glWidget->setLandmarksVisible(visible);
    });
    lmLayout->addWidget(m_landmarksVisibleCheck);

    mainLayout->addWidget(m_landmarksGroup);
    m_landmarksGroup->hide();  // Hidden until landmarks are detected

    // Empty state label
    m_emptyLabel = new QLabel(tr("Load a mesh file to begin"));
    m_emptyLabel->setAlignment(Qt::AlignCenter);
    m_emptyLabel->setStyleSheet("color: gray; font-style: italic;");
    mainLayout->addWidget(m_emptyLabel);

    mainLayout->addStretch();

    m_sidePanel->setWidget(panelWidget);
    addDockWidget(Qt::RightDockWidgetArea, m_sidePanel);
}

void MainWindow::updateLabelCheckboxes()
{
    // Update label names based on jaw type
    QStringList upperLabels = {
        "0: Gingiva",
        "1: UR8 (18)", "2: UR7 (17)", "3: UR6 (16)", "4: UR5 (15)",
        "5: UR4 (14)", "6: UR3 (13)", "7: UR2 (12)", "8: UR1 (11)",
        "9: UL1 (21)", "10: UL2 (22)", "11: UL3 (23)", "12: UL4 (24)",
        "13: UL5 (25)", "14: UL6 (26)", "15: UL7 (27)", "16: UL8 (28)"
    };

    QStringList lowerLabels = {
        "0: Gingiva",
        "1: LR8 (48)", "2: LR7 (47)", "3: LR6 (46)", "4: LR5 (45)",
        "5: LR4 (44)", "6: LR3 (43)", "7: LR2 (42)", "8: LR1 (41)",
        "9: LL1 (31)", "10: LL2 (32)", "11: LL3 (33)", "12: LL4 (34)",
        "13: LL5 (35)", "14: LL6 (36)", "15: LL7 (37)", "16: LL8 (38)"
    };

    const QStringList& labels = m_isUpperJaw ? upperLabels : lowerLabels;

    for (size_t i = 0; i < m_labelCheckboxes.size() && i < labels.size(); ++i) {
        m_labelCheckboxes[i]->setText(labels[i]);
    }
}

void MainWindow::onMeshVisibilityChanged(bool visible)
{
    m_glWidget->setMeshVisible(visible);
}

void MainWindow::onPointCloudVisibilityChanged(bool visible)
{
    m_glWidget->setPointCloudVisible(visible);
}

void MainWindow::onLabelVisibilityChanged(int label, bool visible)
{
    m_glWidget->setLabelVisible(label, visible);
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

    // AI menu
    QMenu* aiMenu = menuBar()->addMenu(tr("&AI"));

    // CrossTooth submenu
    QMenu* crossToothMenu = aiMenu->addMenu(tr("CrossTooth"));
    QAction* segmentAction = crossToothMenu->addAction(tr("&Tooth Segmentation"));
    segmentAction->setShortcut(Qt::CTRL | Qt::Key_S);
    connect(segmentAction, &QAction::triggered, this, &MainWindow::runAISegmentation);

    // 3DTeethLand submenu
    QMenu* teethLandMenu = aiMenu->addMenu(tr("3DTeethLand"));

    QAction* instSegAction = teethLandMenu->addAction(tr("&Instance Segmentation"));
    instSegAction->setShortcut(Qt::CTRL | Qt::Key_I);
    connect(instSegAction, &QAction::triggered, this, &MainWindow::runInstanceSegmentation);

    QAction* landmarkAction = teethLandMenu->addAction(tr("&Landmark Detection"));
    landmarkAction->setShortcut(Qt::CTRL | Qt::Key_L);
    connect(landmarkAction, &QAction::triggered, this, &MainWindow::runLandmarkDetection);

    aiMenu->addSeparator();

    QAction* upperAction = aiMenu->addAction(tr("Upper Jaw Mode"));
    upperAction->setCheckable(true);
    upperAction->setChecked(true);
    connect(upperAction, &QAction::toggled, this, [this](bool checked) {
        m_isUpperJaw = checked;
        updateLabelCheckboxes();
    });

    // View menu
    QMenu* viewMenu = menuBar()->addMenu(tr("&View"));

    QAction* wireframeAction = viewMenu->addAction(tr("&Wireframe"));
    wireframeAction->setCheckable(true);
    wireframeAction->setShortcut(Qt::Key_W);
    connect(wireframeAction, &QAction::toggled, this, &MainWindow::toggleWireframe);

    QAction* resetViewAction = viewMenu->addAction(tr("&Reset View"));
    resetViewAction->setShortcut(Qt::Key_R);
    connect(resetViewAction, &QAction::triggered, this, &MainWindow::resetView);

    viewMenu->addSeparator();

    QAction* sidePanelAction = viewMenu->addAction(tr("Show &Layers Panel"));
    sidePanelAction->setCheckable(true);
    sidePanelAction->setChecked(true);
    connect(sidePanelAction, &QAction::toggled, this, [this](bool checked) {
        m_sidePanel->setVisible(checked);
    });

    // Help menu
    QMenu* helpMenu = menuBar()->addMenu(tr("&Help"));

    QAction* aboutAction = helpMenu->addAction(tr("&About"));
    connect(aboutAction, &QAction::triggered, this, [this]() {
        QMessageBox::about(this, tr("About ToothViewer"),
            tr("ToothViewer v1.0\n\n"
               "3D Mesh Viewer for Dental Applications\n"
               "with AI-powered Tooth Segmentation\n\n"
               "Supports: OBJ, PLY, STL\n\n"
               "Label format: FDI tooth notation"));
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

    QAction* segmentAction = toolbar->addAction(tr("CrossTooth"));
    segmentAction->setToolTip(tr("Run CrossTooth tooth segmentation"));
    connect(segmentAction, &QAction::triggered, this, &MainWindow::runAISegmentation);

    QAction* instSegAction = toolbar->addAction(tr("Instance Seg"));
    instSegAction->setToolTip(tr("Run 3DTeethLand instance segmentation"));
    connect(instSegAction, &QAction::triggered, this, &MainWindow::runInstanceSegmentation);

    QAction* landmarkAction = toolbar->addAction(tr("Landmarks"));
    landmarkAction->setToolTip(tr("Run 3DTeethLand landmark detection"));
    connect(landmarkAction, &QAction::triggered, this, &MainWindow::runLandmarkDetection);

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
    statusBar()->addWidget(m_statusLabel, 1);

    m_progressBar = new QProgressBar();
    m_progressBar->setMaximumWidth(200);
    m_progressBar->setVisible(false);
    statusBar()->addPermanentWidget(m_progressBar);
}

void MainWindow::openFile()
{
    QString filter = tr("3D Mesh Files (*.obj *.ply *.stl);;OBJ Files (*.obj);;PLY Files (*.ply);;STL Files (*.stl);;All Files (*)");
    QString filePath = QFileDialog::getOpenFileName(this, tr("Open Mesh"), QString(), filter);

    if (filePath.isEmpty()) return;

    m_currentFilePath = filePath;

    // Detect upper/lower from filename
    QString fileName = QFileInfo(filePath).fileName().toLower();
    m_isUpperJaw = fileName.contains("upper");
    updateLabelCheckboxes();

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

    // Show mesh controls, hide empty label
    m_emptyLabel->hide();
    m_meshGroup->show();
    m_meshVisibleCheck->setChecked(true);
}

void MainWindow::runAISegmentation()
{
    if (m_currentFilePath.isEmpty()) {
        QMessageBox::warning(this, tr("Error"), tr("Please load a mesh file first."));
        return;
    }

    // Create output path
    QFileInfo fi(m_currentFilePath);
    QString outputPath = fi.absolutePath() + "/" + fi.baseName() + "_segmented.ply";
    m_segmentedFilePath = outputPath;

    m_statusLabel->setText(tr("Running AI segmentation..."));
    m_progressBar->setVisible(true);
    m_progressBar->setRange(0, 0);  // Indeterminate

    m_segmentation->runSegmentationAsync(m_currentFilePath, outputPath, m_isUpperJaw);
}

void MainWindow::onSegmentationFinished(bool success, const QString& outputPath)
{
    m_progressBar->setVisible(false);

    if (success) {
        // Load segmented result as point cloud (keep original mesh)
        auto pointCloud = MeshLoader::load(outputPath.toStdString());
        if (pointCloud) {
            m_glWidget->loadPointCloud(std::move(pointCloud));
            m_statusLabel->setText(tr("Segmentation complete: %1").arg(QFileInfo(outputPath).fileName()));
            setWindowTitle(QString("ToothViewer - %1 (Segmented)")
                          .arg(QFileInfo(m_currentFilePath).fileName()));

            // Create label checkboxes dynamically
            createLabelCheckboxes();

            // Show segmentation controls
            m_segmentationGroup->show();
            m_pointCloudVisibleCheck->setChecked(true);
        } else {
            m_statusLabel->setText(tr("Failed to load segmentation result"));
        }
    } else {
        QString error = m_segmentation->lastError();
        m_statusLabel->setText(tr("Segmentation failed"));
        QMessageBox::warning(this, tr("Segmentation Error"),
                            tr("AI segmentation failed:\n%1").arg(error));
    }
}

void MainWindow::onSegmentationProgress(const QString& message)
{
    m_statusLabel->setText(message);
}

void MainWindow::createLabelCheckboxes()
{
    // Clear existing checkboxes
    for (auto* cb : m_labelCheckboxes) {
        m_labelsLayout->removeWidget(cb);
        delete cb;
    }
    m_labelCheckboxes.clear();

    // Get label names based on jaw type
    QStringList upperLabels = {
        "Gingiva",
        "UR8 (18)", "UR7 (17)", "UR6 (16)", "UR5 (15)",
        "UR4 (14)", "UR3 (13)", "UR2 (12)", "UR1 (11)",
        "UL1 (21)", "UL2 (22)", "UL3 (23)", "UL4 (24)",
        "UL5 (25)", "UL6 (26)", "UL7 (27)", "UL8 (28)"
    };

    QStringList lowerLabels = {
        "Gingiva",
        "LR8 (48)", "LR7 (47)", "LR6 (46)", "LR5 (45)",
        "LR4 (44)", "LR3 (43)", "LR2 (42)", "LR1 (41)",
        "LL1 (31)", "LL2 (32)", "LL3 (33)", "LL4 (34)",
        "LL5 (35)", "LL6 (36)", "LL7 (37)", "LL8 (38)"
    };

    const QStringList& labels = m_isUpperJaw ? upperLabels : lowerLabels;

    // Create checkboxes for labels 0-16
    for (int i = 0; i <= 16; ++i) {
        QCheckBox* cb = new QCheckBox(labels[i]);
        cb->setChecked(true);
        m_labelCheckboxes.push_back(cb);

        connect(cb, &QCheckBox::toggled, this, [this, i](bool checked) {
            onLabelVisibilityChanged(i, checked);
        });

        // Insert before the stretch
        m_labelsLayout->insertWidget(m_labelsLayout->count() - 1, cb);
    }
}

void MainWindow::runInstanceSegmentation()
{
    if (m_currentFilePath.isEmpty()) {
        QMessageBox::warning(this, tr("Error"), tr("Please load a mesh file first."));
        return;
    }

    QFileInfo fi(m_currentFilePath);
    QString outputPath = fi.absolutePath() + "/" + fi.baseName() + "_instances.ply";

    m_statusLabel->setText(tr("Running 3DTeethLand instance segmentation..."));
    m_progressBar->setVisible(true);
    m_progressBar->setRange(0, 0);

    m_segmentation->runTeethLandAsync(m_currentFilePath, outputPath,
                                       Segmentation::TeethLandInstances, m_isUpperJaw);
}

void MainWindow::runLandmarkDetection()
{
    if (m_currentFilePath.isEmpty()) {
        QMessageBox::warning(this, tr("Error"), tr("Please load a mesh file first."));
        return;
    }

    QFileInfo fi(m_currentFilePath);
    QString outputPath = fi.absolutePath() + "/" + fi.baseName() + "_landmarks.json";

    m_statusLabel->setText(tr("Running 3DTeethLand landmark detection..."));
    m_progressBar->setVisible(true);
    m_progressBar->setRange(0, 0);

    m_segmentation->runTeethLandAsync(m_currentFilePath, outputPath,
                                       Segmentation::TeethLandLandmarks, m_isUpperJaw);
}

void MainWindow::onLandmarksFinished(bool success, const QString& jsonPath, const QString& plyPath)
{
    m_progressBar->setVisible(false);

    if (success && !plyPath.isEmpty()) {
        // Load landmarks as point cloud
        auto landmarks = MeshLoader::load(plyPath.toStdString());
        if (landmarks) {
            m_glWidget->loadLandmarks(std::move(landmarks));
            m_statusLabel->setText(tr("Landmarks detected: %1").arg(QFileInfo(jsonPath).fileName()));

            // Show landmarks controls
            m_landmarksGroup->show();
            m_landmarksVisibleCheck->setChecked(true);
        } else {
            m_statusLabel->setText(tr("Failed to load landmarks"));
        }
    } else {
        QString error = m_segmentation->lastError();
        m_statusLabel->setText(tr("Landmark detection failed"));
        QMessageBox::warning(this, tr("Landmark Error"),
                            tr("Landmark detection failed:\n%1").arg(error));
    }
}
