#include "MainWindow.h"
#include "MeshLoader.h"
#include "RLOptimizer.h"
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
#include <QApplication>
#include <QFrame>
#include <QTimer>
#include <QtConcurrent/QtConcurrent>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    setWindowTitle("Bite Finder");
    resize(1400, 900);

    m_segmentation = new Segmentation(this);
    connect(m_segmentation, &Segmentation::segmentationFinished,
            this, &MainWindow::onSegmentationFinished);
    connect(m_segmentation, &Segmentation::segmentationProgress,
            this, &MainWindow::onSegmentationProgress);

    m_biteSimulator = std::make_unique<BiteSimulator>();

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
    connect(m_glWidget, &GLWidget::biteDataLoaded, this, &MainWindow::onBiteDataLoaded);
    connect(m_glWidget, &GLWidget::landmarkPicked, this, &MainWindow::onLandmarkPicked);
    connect(m_glWidget, &GLWidget::landmarkPairComplete, this, &MainWindow::onLandmarkPairComplete);
    connect(m_glWidget, &GLWidget::jawMoved, this, &MainWindow::onJawMoved);
    connect(m_glWidget, &GLWidget::jawMovedFast, this, &MainWindow::onJawMovedFast);
    connect(m_glWidget, &GLWidget::jawSelectionChanged, this, &MainWindow::onJawSelectionChanged);
}

void MainWindow::setupSidePanel()
{
    m_sidePanel = new QDockWidget(tr("Controls"), this);
    m_sidePanel->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable);
    m_sidePanel->setMinimumWidth(250);

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
    m_meshGroup->hide();

    // Point cloud / Segmentation group (initially hidden)
    m_segmentationGroup = new QGroupBox(tr("Segmentation"));
    QVBoxLayout* segLayout = new QVBoxLayout(m_segmentationGroup);

    m_pointCloudVisibleCheck = new QCheckBox(tr("Show All Points"));
    m_pointCloudVisibleCheck->setChecked(true);
    connect(m_pointCloudVisibleCheck, &QCheckBox::toggled, this, &MainWindow::onPointCloudVisibilityChanged);
    segLayout->addWidget(m_pointCloudVisibleCheck);

    QHBoxLayout* buttonLayout = new QHBoxLayout();
    QPushButton* selectAllBtn = new QPushButton(tr("All"));
    QPushButton* selectNoneBtn = new QPushButton(tr("None"));
    buttonLayout->addWidget(selectAllBtn);
    buttonLayout->addWidget(selectNoneBtn);
    segLayout->addLayout(buttonLayout);

    connect(selectAllBtn, &QPushButton::clicked, this, [this]() {
        for (auto* cb : m_labelCheckboxes) cb->setChecked(true);
    });
    connect(selectNoneBtn, &QPushButton::clicked, this, [this]() {
        for (auto* cb : m_labelCheckboxes) cb->setChecked(false);
    });

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
    m_segmentationGroup->hide();

    m_biteGroup = new QGroupBox(tr("Bite Optimizer"));
    QVBoxLayout* biteLayout = new QVBoxLayout(m_biteGroup);
    biteLayout->setSpacing(6);

    QHBoxLayout* visLayout = new QHBoxLayout();
    m_maxillaVisibleCheck = new QCheckBox(tr("Upper"));
    m_maxillaVisibleCheck->setChecked(true);
    connect(m_maxillaVisibleCheck, &QCheckBox::toggled, this, &MainWindow::onMaxillaVisibilityChanged);
    visLayout->addWidget(m_maxillaVisibleCheck);
    m_mandibleVisibleCheck = new QCheckBox(tr("Lower"));
    m_mandibleVisibleCheck->setChecked(true);
    connect(m_mandibleVisibleCheck, &QCheckBox::toggled, this, &MainWindow::onMandibleVisibilityChanged);
    visLayout->addWidget(m_mandibleVisibleCheck);
    m_contactPointsCheck = new QCheckBox(tr("Contact"));
    m_contactPointsCheck->setChecked(true);
    connect(m_contactPointsCheck, &QCheckBox::toggled, this, &MainWindow::onContactPointsVisibilityChanged);
    visLayout->addWidget(m_contactPointsCheck);
    biteLayout->addLayout(visLayout);

    m_optimizeBtn = new QPushButton(tr("Optimize Bite"));
    m_optimizeBtn->setStyleSheet("QPushButton { padding: 12px; font-weight: bold; font-size: 14px; background-color: #4CAF50; color: white; }");
    connect(m_optimizeBtn, &QPushButton::clicked, this, &MainWindow::runOptimizeBite);
    biteLayout->addWidget(m_optimizeBtn);

    m_stopBtn = new QPushButton(tr("Stop"));
    m_stopBtn->setStyleSheet("QPushButton { padding: 8px; background-color: #f44336; color: white; }");
    m_stopBtn->hide();
    connect(m_stopBtn, &QPushButton::clicked, this, &MainWindow::stopOptimization);
    biteLayout->addWidget(m_stopBtn);

    m_biteProgressBar = new QProgressBar();
    m_biteProgressBar->setTextVisible(true);
    m_biteProgressBar->hide();
    biteLayout->addWidget(m_biteProgressBar);

    QHBoxLayout* actionLayout = new QHBoxLayout();
    m_resetBtn = new QPushButton(tr("Reset"));
    connect(m_resetBtn, &QPushButton::clicked, this, &MainWindow::resetBiteAlignment);
    actionLayout->addWidget(m_resetBtn);
    m_exportBtn = new QPushButton(tr("Export"));
    connect(m_exportBtn, &QPushButton::clicked, this, &MainWindow::exportAlignedMeshes);
    actionLayout->addWidget(m_exportBtn);
    biteLayout->addLayout(actionLayout);

    m_metricsLabel = new QLabel();
    m_metricsLabel->setWordWrap(true);
    m_metricsLabel->setStyleSheet("QLabel { background-color: #f5f5f5; padding: 10px; border-radius: 4px; font-size: 12px; }");
    biteLayout->addWidget(m_metricsLabel);

    m_maxillaSegCheck = nullptr;
    m_mandibleSegCheck = nullptr;
    m_fdiLabelsMaxillaCheck = nullptr;
    m_fdiLabelsMandibleCheck = nullptr;
    m_toothAxesMaxillaCheck = nullptr;
    m_toothAxesMandibleCheck = nullptr;
    m_roughAlignBtn = nullptr;
    m_landmarkBtn = nullptr;
    m_landmarkWidget = nullptr;
    m_landmarkInstructionLabel = nullptr;
    m_landmarkApplyBtn = nullptr;
    m_landmarkCancelBtn = nullptr;
    m_quickBiteBtn = nullptr;
    m_cemBtn = nullptr;
    m_esBtn = nullptr;
    m_ppoBtn = nullptr;
    m_phaseLabel = nullptr;
    m_phase1Label = nullptr;
    m_phase2Label = nullptr;
    m_phase3Label = nullptr;
    m_beforeAfterCheck = nullptr;
    m_movementGroup = nullptr;
    m_moveMaxillaCheck = nullptr;
    m_keyboardHelpLabel = nullptr;
    m_applyManualBtn = nullptr;
    m_sliderX = nullptr; m_sliderY = nullptr; m_sliderZ = nullptr;
    m_sliderRotX = nullptr; m_sliderRotY = nullptr; m_sliderRotZ = nullptr;
    m_labelX = nullptr; m_labelY = nullptr; m_labelZ = nullptr;
    m_labelRotX = nullptr; m_labelRotY = nullptr; m_labelRotZ = nullptr;

    mainLayout->addWidget(m_biteGroup);
    m_biteGroup->hide();

    // Empty state label
    m_emptyLabel = new QLabel(tr("Load upper/lower jaw meshes\nto begin bite optimization"));
    m_emptyLabel->setAlignment(Qt::AlignCenter);
    m_emptyLabel->setStyleSheet("color: gray; font-style: italic; font-size: 14px;");
    mainLayout->addWidget(m_emptyLabel);

    mainLayout->addStretch();

    m_sidePanel->setWidget(panelWidget);
    addDockWidget(Qt::RightDockWidgetArea, m_sidePanel);
}

void MainWindow::updateLabelCheckboxes()
{
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
    QMenu* fileMenu = menuBar()->addMenu(tr("&File"));

    QAction* openAction = fileMenu->addAction(tr("&Open Mesh..."));
    openAction->setShortcut(QKeySequence::Open);
    connect(openAction, &QAction::triggered, this, &MainWindow::openFile);

    QAction* loadBiteAction = fileMenu->addAction(tr("&Load Bite Data..."));
    loadBiteAction->setShortcut(Qt::CTRL | Qt::Key_L);
    connect(loadBiteAction, &QAction::triggered, this, &MainWindow::loadBiteData);

    fileMenu->addSeparator();

    QAction* saveAction = fileMenu->addAction(tr("&Save As..."));
    saveAction->setShortcut(QKeySequence::SaveAs);
    connect(saveAction, &QAction::triggered, this, &MainWindow::saveFile);

    QAction* exportAction = fileMenu->addAction(tr("&Export Aligned Meshes..."));
    connect(exportAction, &QAction::triggered, this, &MainWindow::exportAlignedMeshes);

    fileMenu->addSeparator();

    QAction* exitAction = fileMenu->addAction(tr("E&xit"));
    exitAction->setShortcut(QKeySequence::Quit);
    connect(exitAction, &QAction::triggered, this, &QMainWindow::close);

    QMenu* aiMenu = menuBar()->addMenu(tr("&AI"));

    QAction* segmentAction = aiMenu->addAction(tr("&Tooth Segmentation (Unavailable)"));
    segmentAction->setShortcut(Qt::CTRL | Qt::Key_T);
    segmentAction->setEnabled(false);
    segmentAction->setToolTip(tr("AI segmentation requires Python environment (not installed)"));

    aiMenu->addSeparator();

    QAction* upperAction = aiMenu->addAction(tr("Upper Jaw Mode"));
    upperAction->setCheckable(true);
    upperAction->setChecked(true);
    connect(upperAction, &QAction::toggled, this, [this](bool checked) {
        m_isUpperJaw = checked;
        updateLabelCheckboxes();
    });

    // Bite menu
    QMenu* biteMenu = menuBar()->addMenu(tr("&Bite"));

    QAction* quickBiteAction = biteMenu->addAction(tr("&Quick Bite (Preview)"));
    quickBiteAction->setShortcut(Qt::CTRL | Qt::Key_B);
    connect(quickBiteAction, &QAction::triggered, this, &MainWindow::runQuickBite);

    QAction* optimizeAction = biteMenu->addAction(tr("&Find Best Occlusion"));
    optimizeAction->setShortcut(Qt::CTRL | Qt::Key_O);
    connect(optimizeAction, &QAction::triggered, this, &MainWindow::runOptimizeBite);

    biteMenu->addSeparator();

    QAction* resetBiteAction = biteMenu->addAction(tr("&Reset Alignment"));
    connect(resetBiteAction, &QAction::triggered, this, &MainWindow::resetBiteAlignment);

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

    QAction* sidePanelAction = viewMenu->addAction(tr("Show &Controls Panel"));
    sidePanelAction->setCheckable(true);
    sidePanelAction->setChecked(true);
    connect(sidePanelAction, &QAction::toggled, this, [this](bool checked) {
        m_sidePanel->setVisible(checked);
    });

    // Help menu
    QMenu* helpMenu = menuBar()->addMenu(tr("&Help"));

    QAction* aboutAction = helpMenu->addAction(tr("&About"));
    connect(aboutAction, &QAction::triggered, this, [this]() {
        QMessageBox::about(this, tr("About Bite Finder"),
            tr("Bite Finder v1.0\n\n"
               "AI-Powered Dental Occlusion Optimizer\n\n"
               "Features:\n"
               "- Automatic bite alignment\n"
               "- Contact optimization\n"
               "- Real-time visualization\n\n"
               "Supports: OBJ, PLY, STL"));
    });
}

void MainWindow::setupToolBar()
{
    QToolBar* toolbar = addToolBar(tr("Main Toolbar"));
    toolbar->setMovable(false);

    QAction* loadBiteAction = toolbar->addAction(tr("Load Bite"));
    loadBiteAction->setToolTip(tr("Load upper and lower jaw meshes"));
    connect(loadBiteAction, &QAction::triggered, this, &MainWindow::loadBiteData);

    QAction* loadExampleAction = toolbar->addAction(tr("Example"));
    loadExampleAction->setToolTip(tr("Load example jaw meshes"));
    connect(loadExampleAction, &QAction::triggered, this, &MainWindow::loadExampleData);

    QAction* aiSegmentAction = toolbar->addAction(tr("AI Segment"));
    aiSegmentAction->setToolTip(tr("AI segmentation requires Python environment (not installed)"));
    aiSegmentAction->setEnabled(false);

    toolbar->addSeparator();

    QAction* quickBiteAction = toolbar->addAction(tr("Quick Bite"));
    quickBiteAction->setToolTip(tr("Fast initial alignment"));
    connect(quickBiteAction, &QAction::triggered, this, &MainWindow::runQuickBite);

    QAction* optimizeAction = toolbar->addAction(tr("Optimize"));
    optimizeAction->setToolTip(tr("Find best occlusion"));
    connect(optimizeAction, &QAction::triggered, this, &MainWindow::runOptimizeBite);

    toolbar->addSeparator();

    QAction* resetAction = toolbar->addAction(tr("Reset View"));
    connect(resetAction, &QAction::triggered, this, &MainWindow::resetView);

    QAction* wireframeAction = toolbar->addAction(tr("Wireframe"));
    wireframeAction->setCheckable(true);
    connect(wireframeAction, &QAction::toggled, this, &MainWindow::toggleWireframe);
}

void MainWindow::setupStatusBar()
{
    m_statusLabel = new QLabel(tr("Load bite data to begin"));
    statusBar()->addWidget(m_statusLabel, 1);

    m_progressBar = new QProgressBar();
    m_progressBar->setMaximumWidth(200);
    m_progressBar->setVisible(false);
    statusBar()->addPermanentWidget(m_progressBar);
}

void MainWindow::openFile()
{
    QString filter = tr("3D Mesh Files (*.obj *.ply *.stl);;All Files (*)");
    QString filePath = QFileDialog::getOpenFileName(this, tr("Open Mesh"), QString(), filter);

    if (filePath.isEmpty()) return;

    m_currentFilePath = filePath;

    QString fileName = QFileInfo(filePath).fileName().toLower();
    m_isUpperJaw = fileName.contains("upper");
    updateLabelCheckboxes();

    auto mesh = MeshLoader::load(filePath.toStdString());
    if (mesh) {
        m_glWidget->loadMesh(std::move(mesh));
        setWindowTitle(QString("Bite Finder - %1").arg(QFileInfo(filePath).fileName()));
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
    m_statusLabel->setText(QString("Vertices: %1 | Faces: %2").arg(vertices).arg(faces));
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

    QFileInfo fi(m_currentFilePath);
    QString outputPath = fi.absolutePath() + "/" + fi.baseName() + "_segmented.ply";
    m_segmentedFilePath = outputPath;

    m_statusLabel->setText(tr("Running AI segmentation..."));
    m_progressBar->setVisible(true);
    m_progressBar->setRange(0, 0);

    m_segmentation->runSegmentationAsync(m_currentFilePath, outputPath, m_isUpperJaw);
}

void MainWindow::onSegmentationFinished(bool success, const QString& outputPath)
{
    m_progressBar->setVisible(false);

    if (success) {
        auto pointCloud = MeshLoader::load(outputPath.toStdString());
        if (pointCloud) {
            m_glWidget->loadPointCloud(std::move(pointCloud));
            m_statusLabel->setText(tr("Segmentation complete"));
            createLabelCheckboxes();
            m_segmentationGroup->show();
            m_pointCloudVisibleCheck->setChecked(true);
        }
    } else {
        QString error = m_segmentation->lastError();
        m_statusLabel->setText(tr("Segmentation failed"));
        QMessageBox::warning(this, tr("Error"), tr("AI segmentation failed:\n%1").arg(error));
    }
}

void MainWindow::onSegmentationProgress(const QString& message)
{
    m_statusLabel->setText(message);
}

void MainWindow::runBiteSegmentation()
{
    if (m_maxillaPath.isEmpty() || m_mandiblePath.isEmpty()) {
        QMessageBox::warning(this, tr("Error"), tr("Please load bite data first."));
        return;
    }

    m_statusLabel->setText(tr("Segmenting upper jaw (maxilla)..."));
    m_progressBar->setVisible(true);
    m_progressBar->setRange(0, 0);

    m_biteSegState = BiteSegState::SegmentingMaxilla;

    QFileInfo fi(m_maxillaPath);
    QString outputPath = fi.absolutePath() + "/" + fi.baseName() + "_segmented.ply";

    disconnect(m_segmentation, &Segmentation::segmentationFinished, this, &MainWindow::onSegmentationFinished);
    connect(m_segmentation, &Segmentation::segmentationFinished, this, &MainWindow::onBiteSegmentationStep, Qt::UniqueConnection);

    m_segmentation->runSegmentationAsync(m_maxillaPath, outputPath, true);
}

void MainWindow::onBiteSegmentationStep()
{
    if (m_biteSegState == BiteSegState::SegmentingMaxilla) {
        m_statusLabel->setText(tr("Maxilla done! Starting mandible..."));
        m_biteSegState = BiteSegState::SegmentingMandible;

        QFileInfo fiMand(m_mandiblePath);
        QString outputPath = fiMand.absolutePath() + "/" + fiMand.baseName() + "_segmented.ply";

        // 딜레이 후 다음 세그먼테이션 시작 (시그널 핸들러에서 바로 하면 크래시)
        QTimer::singleShot(100, this, [this, outputPath]() {
            m_segmentation->runSegmentationAsync(m_mandiblePath, outputPath, false);
        });

    } else if (m_biteSegState == BiteSegState::SegmentingMandible) {
        m_statusLabel->setText(tr("Segmentation complete!"));
        m_biteSegState = BiteSegState::Done;
        m_progressBar->setVisible(false);

        disconnect(m_segmentation, &Segmentation::segmentationFinished, this, &MainWindow::onBiteSegmentationStep);
        connect(m_segmentation, &Segmentation::segmentationFinished, this, &MainWindow::onSegmentationFinished);

        // 딜레이 후 상/하악 결과 모두 로드
        QTimer::singleShot(200, this, [this]() {
            // 상악 세그먼테이션 로드
            QFileInfo fiMax(m_maxillaPath);
            QString maxSegPath = fiMax.absolutePath() + "/" + fiMax.baseName() + "_segmented.ply";
            auto maxSeg = MeshLoader::load(maxSegPath.toStdString());
            if (maxSeg) {
                m_glWidget->loadMaxillaSegmentation(std::move(maxSeg));
            }

            // 하악 세그먼테이션 로드
            QFileInfo fiMand(m_mandiblePath);
            QString mandSegPath = fiMand.absolutePath() + "/" + fiMand.baseName() + "_segmented.ply";
            auto mandSeg = MeshLoader::load(mandSegPath.toStdString());
            if (mandSeg) {
                m_glWidget->loadMandibleSegmentation(std::move(mandSeg));
            }

            m_statusLabel->setText(tr("Segmentation complete! (Upper & Lower)"));
            updateMetricsDisplay();
        });
    }
}

void MainWindow::createLabelCheckboxes()
{
    for (auto* cb : m_labelCheckboxes) {
        m_labelsLayout->removeWidget(cb);
        delete cb;
    }
    m_labelCheckboxes.clear();

    QStringList upperLabels = {
        "Gingiva", "UR8", "UR7", "UR6", "UR5", "UR4", "UR3", "UR2", "UR1",
        "UL1", "UL2", "UL3", "UL4", "UL5", "UL6", "UL7", "UL8"
    };

    QStringList lowerLabels = {
        "Gingiva", "LR8", "LR7", "LR6", "LR5", "LR4", "LR3", "LR2", "LR1",
        "LL1", "LL2", "LL3", "LL4", "LL5", "LL6", "LL7", "LL8"
    };

    const QStringList& labels = m_isUpperJaw ? upperLabels : lowerLabels;

    for (int i = 0; i <= 16; ++i) {
        QCheckBox* cb = new QCheckBox(labels[i]);
        cb->setChecked(true);
        m_labelCheckboxes.push_back(cb);
        connect(cb, &QCheckBox::toggled, this, [this, i](bool checked) {
            onLabelVisibilityChanged(i, checked);
        });
        m_labelsLayout->insertWidget(m_labelsLayout->count() - 1, cb);
    }
}

// ========== Bite Optimization Methods ==========

void MainWindow::loadExampleData()
{
    QString appDir = QCoreApplication::applicationDirPath();
    
#ifdef Q_OS_MAC
    QString assetsDir = appDir + "/../../../../assets";
#else
    QString assetsDir = appDir + "/assets";
#endif
    
    QString maxillaPath = assetsDir + "/ZOUIF2W4_upper.obj";
    QString mandiblePath = assetsDir + "/ZOUIF2W4_lower.obj";
    
    if (!QFile::exists(maxillaPath) || !QFile::exists(mandiblePath)) {
        QMessageBox::warning(this, tr("Error"), 
            tr("Example files not found.\nExpected at: %1").arg(assetsDir));
        return;
    }
    
    m_maxillaPath = maxillaPath;
    m_mandiblePath = mandiblePath;
    
    m_statusLabel->setText(tr("Loading example upper jaw..."));
    QApplication::processEvents();
    
    if (!m_biteSimulator->loadMaxilla(maxillaPath.toStdString())) {
        QMessageBox::warning(this, tr("Error"), tr("Failed to load upper jaw mesh."));
        return;
    }
    
    m_statusLabel->setText(tr("Loading example lower jaw..."));
    QApplication::processEvents();
    
    if (!m_biteSimulator->loadMandible(mandiblePath.toStdString())) {
        QMessageBox::warning(this, tr("Error"), tr("Failed to load lower jaw mesh."));
        return;
    }
    
    auto maxillaMesh = MeshLoader::load(maxillaPath.toStdString());
    auto mandibleMesh = MeshLoader::load(mandiblePath.toStdString());
    
    if (maxillaMesh && mandibleMesh) {
        m_glWidget->loadMaxilla(std::move(maxillaMesh));
        m_glWidget->loadMandible(std::move(mandibleMesh));
        
        setWindowTitle(tr("Bite Finder - Example Data"));
        m_statusLabel->setText(tr("Example loaded - Click 'Optimize' to find best occlusion"));
    }
}

void MainWindow::loadBiteData()
{
    // Step 1: Load Upper Jaw
    QMessageBox::information(this, tr("Step 1 of 2"),
        tr("First, select the Upper Jaw (Maxilla) mesh file."));

    QString maxillaPath = QFileDialog::getOpenFileName(
        this, tr("Step 1/2: Select Upper Jaw (Maxilla)"), QString(),
        tr("3D Mesh Files (*.obj *.ply *.stl);;All Files (*)"));

    if (maxillaPath.isEmpty()) {
        m_statusLabel->setText(tr("Load cancelled"));
        return;
    }

    // Step 2: Load Lower Jaw
    QMessageBox::information(this, tr("Step 2 of 2"),
        tr("Upper jaw selected.\n\nNow select the Lower Jaw (Mandible) mesh file."));

    QString mandiblePath = QFileDialog::getOpenFileName(
        this, tr("Step 2/2: Select Lower Jaw (Mandible)"), QFileInfo(maxillaPath).absolutePath(),
        tr("3D Mesh Files (*.obj *.ply *.stl);;All Files (*)"));

    if (mandiblePath.isEmpty()) {
        m_statusLabel->setText(tr("Load cancelled"));
        return;
    }

    m_maxillaPath = maxillaPath;
    m_mandiblePath = mandiblePath;

    m_statusLabel->setText(tr("Loading upper jaw..."));
    QApplication::processEvents();

    if (!m_biteSimulator->loadMaxilla(maxillaPath.toStdString())) {
        QMessageBox::warning(this, tr("Error"), tr("Failed to load upper jaw mesh."));
        return;
    }

    m_statusLabel->setText(tr("Loading lower jaw..."));
    QApplication::processEvents();

    if (!m_biteSimulator->loadMandible(mandiblePath.toStdString())) {
        QMessageBox::warning(this, tr("Error"), tr("Failed to load lower jaw mesh."));
        return;
    }

    auto maxillaMesh = MeshLoader::load(maxillaPath.toStdString());
    auto mandibleMesh = MeshLoader::load(mandiblePath.toStdString());

    if (maxillaMesh && mandibleMesh) {
        m_glWidget->loadMaxilla(std::move(maxillaMesh));
        m_glWidget->loadMandible(std::move(mandibleMesh));

        setWindowTitle(QString("Bite Finder - %1 / %2")
                      .arg(QFileInfo(maxillaPath).fileName())
                      .arg(QFileInfo(mandiblePath).fileName()));

        m_statusLabel->setText(tr("Ready - Click 'Quick Bite' to begin"));
    }
}

void MainWindow::onBiteDataLoaded()
{
    m_emptyLabel->hide();
    m_biteGroup->show();

    // Set up BiteSimulator reference for keyboard control
    m_glWidget->setBiteSimulator(m_biteSimulator.get());
    m_glWidget->setFocus();  // Enable keyboard input

    // Calculate initial metrics
    OrthodonticMetrics metrics = m_biteSimulator->computeMetrics();
    m_initialReward = m_biteSimulator->computeReward(metrics);
    m_previousReward = m_initialReward;

    updateContactPointsVisualization();
    updateMetricsDisplay();
}

void MainWindow::roughAlignJaws()
{
    if (!m_biteSimulator->maxilla() || !m_biteSimulator->mandible()) {
        QMessageBox::warning(this, tr("Error"), tr("Please load bite data first."));
        return;
    }

    m_statusLabel->setText(tr("Rough aligning..."));
    m_roughAlignBtn->setEnabled(false);
    QApplication::processEvents();

    m_biteSimulator->roughAlign();

    // Update the mandible mesh in the viewer
    m_glWidget->updateMandibleFromSimulator(m_biteSimulator->mandible());

    // Re-adjust camera to fit both meshes after alignment
    m_glWidget->resetView();

    updateContactPointsVisualization();
    updateMetricsDisplay();

    m_roughAlignBtn->setEnabled(true);
    m_statusLabel->setText(tr("Rough alignment complete - Meshes positioned"));
}

void MainWindow::runQuickBite()
{
    if (!m_biteSimulator->maxilla() || !m_biteSimulator->mandible()) {
        QMessageBox::warning(this, tr("Error"), tr("Please load bite data first."));
        return;
    }

    m_statusLabel->setText(tr("Aligning..."));
    m_quickBiteBtn->setEnabled(false);
    QApplication::processEvents();

    m_biteSimulator->runICPAlignment(30);

    m_glWidget->updateMandibleFromSimulator(m_biteSimulator->mandible());
    updateContactPointsVisualization();

    updateMetricsDisplay();
    m_quickBiteBtn->setEnabled(true);
    m_statusLabel->setText(tr("Initial alignment complete - Click 'Find Best Occlusion' to optimize"));
}

void MainWindow::runOptimizeBite()
{
    if (!m_biteSimulator->maxilla() || !m_biteSimulator->mandible()) {
        QMessageBox::warning(this, tr("Error"), tr("Please load bite data first."));
        return;
    }

    m_optimizeBtn->setEnabled(false);
    m_stopBtn->show();
    m_biteProgressBar->show();
    m_biteProgressBar->setRange(0, 100);
    m_biteProgressBar->setValue(0);
    m_optimizationRunning = true;
    m_statusLabel->setText(tr("Optimizing..."));

    m_optStep = 0;
    m_optBestReward = -999;
    
    auto future = QtConcurrent::run([this]() {
        Eigen::Matrix4f initialTransform = m_biteSimulator->transformMatrix();
        Eigen::Matrix4f bestTransform = initialTransform;
        
        m_biteSimulator->runICPAlignment(15);
        m_biteSimulator->cacheSamplePoints();
        m_optBestReward = m_biteSimulator->computeFastReward();
        bestTransform = m_biteSimulator->transformMatrix();
        m_optStep = 5;
        
        const int numStarts = 5;
        const int stepsPerStart = 40;
        
        for (int start = 0; start < numStarts && m_optimizationRunning; ++start) {
            m_biteSimulator->reset();
            m_biteSimulator->setMandibleTransform(bestTransform);
            
            if (start > 0) {
                float rx = (rand() / (float)RAND_MAX - 0.5f) * 4.0f;
                float ry = (rand() / (float)RAND_MAX - 0.5f) * 4.0f;
                float rz = (rand() / (float)RAND_MAX - 0.5f) * 4.0f;
                float tx = (rand() / (float)RAND_MAX - 0.5f) * 2.0f;
                float ty = (rand() / (float)RAND_MAX - 0.5f) * 2.0f;
                float tz = (rand() / (float)RAND_MAX - 0.5f) * 1.0f;
                m_biteSimulator->applyTransform(Eigen::Vector3f(rx, ry, rz), Eigen::Vector3f(tx, ty, tz));
            }
            
            m_biteSimulator->cacheSamplePoints();
            
            float lr = 0.15f;
            for (int i = 0; i < stepsPerStart && m_optimizationRunning; ++i) {
                if (i > stepsPerStart / 2) lr = 0.05f;
                double reward = m_biteSimulator->runAdamStep(lr);
                if (reward > m_optBestReward) {
                    m_optBestReward = reward;
                    bestTransform = m_biteSimulator->transformMatrix();
                }
                m_optStep = 5 + start * stepsPerStart / numStarts + i * 95 / (numStarts * stepsPerStart);
            }
        }
        
        m_biteSimulator->reset();
        m_biteSimulator->setMandibleTransform(bestTransform);
        m_biteSimulator->cacheSamplePoints();
        return m_optBestReward;
    });

    if (!m_optimizationWatcher) {
        m_optimizationWatcher = new QFutureWatcher<double>(this);
        connect(m_optimizationWatcher, &QFutureWatcher<double>::finished, this, [this]() {
            m_glWidget->updateMandibleFromSimulator(m_biteSimulator->mandible());
            m_glWidget->update();
            onOptimizationFinished();
        });
    }
    m_optimizationWatcher->setFuture(future);

    m_optTimer = new QTimer(this);
    connect(m_optTimer, &QTimer::timeout, this, [this]() {
        if (!m_optimizationRunning) {
            m_optTimer->stop();
            m_optTimer->deleteLater();
            m_optTimer = nullptr;
            return;
        }
        m_biteProgressBar->setValue(m_optStep);
        m_statusLabel->setText(tr("Score: %1%").arg((m_optBestReward + 1.0) / 2.0 * 100.0, 0, 'f', 1));
    });
    m_optTimer->start(50);
}

void MainWindow::runOptimizationStep()
{
}

void MainWindow::stopOptimization()
{
    if (m_optimizationRunning) {
        m_optimizationRunning = false;
        if (m_optTimer) { m_optTimer->stop(); m_optTimer->deleteLater(); m_optTimer = nullptr; }
        if (m_optimizationWatcher) m_optimizationWatcher->cancel();
        
        m_biteSimulator->cancelOptimization();
        
        m_quickBiteBtn->setEnabled(true);
        m_optimizeBtn->setEnabled(true);
        m_cemBtn->setEnabled(true);
        m_stopBtn->hide();
        m_biteProgressBar->hide();
        
        m_glWidget->updateMandibleFromSimulator(m_biteSimulator->mandible());
        m_glWidget->update();
        updateMetricsDisplay();
        
        m_statusLabel->setText(tr("Stopped: %1%").arg((m_optBestReward + 1.0) / 2.0 * 100.0, 0, 'f', 1));
    }
}

void MainWindow::onOptimizationFinished()
{
    m_optimizationRunning = false;
    m_optimizeBtn->setEnabled(true);
    m_cemBtn->setEnabled(true);
    m_stopBtn->hide();
    m_biteProgressBar->hide();
    
    m_glWidget->updateMandibleFromSimulator(m_biteSimulator->mandible());
    updateContactPointsVisualization();
    updateMetricsDisplay();
    
    double improvement = m_optBestReward - m_initialReward;
    m_statusLabel->setText(tr("Optimization complete! Score: %1% (improvement: %2%)")
        .arg((m_optBestReward + 1.0) / 2.0 * 100.0, 0, 'f', 1)
        .arg(improvement * 50.0, 0, 'f', 1));
}

void MainWindow::setOptimizationPhase(int phase)
{
    QString activeStyle = "color: #4CAF50; font-weight: bold;";
    QString completeStyle = "color: #2196F3;";
    QString inactiveStyle = "color: gray;";

    m_phase1Label->setStyleSheet(phase == 1 ? activeStyle : (phase > 1 ? completeStyle : inactiveStyle));
    m_phase2Label->setStyleSheet(phase == 2 ? activeStyle : (phase > 2 ? completeStyle : inactiveStyle));
    m_phase3Label->setStyleSheet(phase == 3 ? activeStyle : inactiveStyle);

    // Update phase labels with checkmarks
    m_phase1Label->setText(phase > 1 ? tr("  Phase 1: Stabilizing contact  ") : tr("  Phase 1: Stabilizing contact"));
    m_phase2Label->setText(phase > 2 ? tr("  Phase 2: Balancing occlusion  ") : tr("  Phase 2: Balancing occlusion"));
    m_phase3Label->setText(tr("  Phase 3: Fine adjustment"));
}

void MainWindow::resetBiteAlignment()
{
    if (!m_biteSimulator->mandible()) return;

    m_biteSimulator->reset();

    if (!m_mandiblePath.isEmpty()) {
        auto mandibleMesh = MeshLoader::load(m_mandiblePath.toStdString());
        if (mandibleMesh) {
            m_glWidget->loadMandible(std::move(mandibleMesh));
        }
    }

    m_beforeAfterCheck->setChecked(false);
    updateMetricsDisplay();
    m_statusLabel->setText(tr("Reset to initial position"));
}

void MainWindow::toggleBeforeAfter(bool showBefore)
{
    if (!m_biteSimulator->mandible()) return;

    m_showingBeforeState = showBefore;

    if (showBefore) {
        // Show initial state
        if (!m_mandiblePath.isEmpty()) {
            auto mandibleMesh = MeshLoader::load(m_mandiblePath.toStdString());
            if (mandibleMesh) {
                m_glWidget->loadMandible(std::move(mandibleMesh));
            }
        }
        m_statusLabel->setText(tr("Showing: Before alignment"));
    } else {
        // Show current optimized state
        m_glWidget->updateMandibleFromSimulator(m_biteSimulator->mandible());
        m_statusLabel->setText(tr("Showing: After alignment"));
    }
}

void MainWindow::exportAlignedMeshes()
{
    if (!m_biteSimulator->mandible()) {
        QMessageBox::warning(this, tr("Error"), tr("No aligned meshes to export."));
        return;
    }

    QString dir = QFileDialog::getExistingDirectory(this, tr("Select Export Directory"));
    if (dir.isEmpty()) return;

    QString upperPath = dir + "/upper_aligned.obj";
    QString lowerPath = dir + "/lower_aligned.obj";

    bool upperSaved = MeshLoader::save(*m_biteSimulator->maxilla(), upperPath.toStdString());
    bool lowerSaved = m_biteSimulator->saveMandible(lowerPath.toStdString());

    if (upperSaved && lowerSaved) {
        m_statusLabel->setText(tr("Exported to: %1").arg(dir));
        QMessageBox::information(this, tr("Export Complete"),
            tr("Aligned meshes exported successfully:\n- upper_aligned.obj\n- lower_aligned.obj"));
    } else {
        QMessageBox::warning(this, tr("Error"), tr("Failed to export some meshes."));
    }
}

void MainWindow::updateMetricsDisplay()
{
    if (!m_biteSimulator->maxilla() || !m_biteSimulator->mandible()) {
        m_metricsLabel->setText(tr("Load data to see metrics"));
        return;
    }

    OrthodonticMetrics m = m_biteSimulator->computeMetrics();
    double score = (m_biteSimulator->computeReward(m) + 1.0) / 2.0 * 100.0;

    QString scoreColor = score >= 70 ? "#4CAF50" : (score >= 50 ? "#FFC107" : "#F44336");
    
    QString text = QString(
        "<div style='text-align:center;'>"
        "<span style='font-size:24px; font-weight:bold; color:%1;'>%2%</span><br>"
        "<span style='color:gray;'>Occlusion Score</span></div><br>"
        "<table width='100%'>"
        "<tr><td>Contacts</td><td align='right'><b>%3</b></td></tr>"
        "<tr><td>Penetration</td><td align='right' style='color:%4;'><b>%5</b></td></tr>"
        "<tr><td>L/R Balance</td><td align='right'>%6 / %7</td></tr>"
        "<tr><td>Molar Load</td><td align='right'>%8%</td></tr>"
        "</table>")
        .arg(scoreColor)
        .arg(score, 0, 'f', 1)
        .arg(m.contact_point_count)
        .arg(m.penetration_count > 10 ? "#F44336" : "#4CAF50")
        .arg(static_cast<int>(m.penetration_count))
        .arg(m.force_left, 0, 'f', 1)
        .arg(m.force_right, 0, 'f', 1)
        .arg(m.protection_ratio * 100, 0, 'f', 0);

    m_metricsLabel->setText(text);
}

void MainWindow::onMaxillaVisibilityChanged(bool visible)
{
    m_glWidget->setMaxillaVisible(visible);
}

void MainWindow::onMandibleVisibilityChanged(bool visible)
{
    m_glWidget->setMandibleVisible(visible);
}

void MainWindow::onContactPointsVisibilityChanged(bool visible)
{
    Q_UNUSED(visible);
    updateContactPointsVisualization();
}

void MainWindow::updateContactPointsVisualization()
{
    if (!m_biteSimulator->maxilla() || !m_biteSimulator->mandible()) return;

    if (m_contactPointsCheck->isChecked()) {
        // Use mesh coloring for contact visualization
        auto contactColors = m_biteSimulator->computeContactColors(1.0f);
        m_glWidget->updateMandibleColors(contactColors);
    } else {
        // Reset to default mandible color (light blue)
        std::vector<Eigen::Vector3f> defaultColor(
            m_glWidget->mandible() ? m_glWidget->mandible()->vertexCount() : 0,
            Eigen::Vector3f(0.68f, 0.85f, 0.90f));
        m_glWidget->updateMandibleColors(defaultColor);
    }
}

// ========== Landmark Alignment Methods ==========

void MainWindow::startLandmarkPicking()
{
    if (!m_biteSimulator->maxilla() || !m_biteSimulator->mandible()) {
        QMessageBox::warning(this, tr("Error"), tr("Please load bite data first."));
        return;
    }

    // Enable landmark picking mode
    m_glWidget->setLandmarkPickingMode(true);

    // Update UI
    m_landmarkBtn->hide();
    m_roughAlignBtn->setEnabled(false);
    m_quickBiteBtn->setEnabled(false);
    m_optimizeBtn->setEnabled(false);
    m_landmarkWidget->show();

    // Show initial instruction
    m_landmarkInstructionLabel->setText(
        tr("<b>Step 1:</b> Click on the <span style='color:#E65100'>UPPER JAW</span> (beige)\n"
           "Pick a recognizable point like a cusp tip.\n\n"
           "<i>Pairs picked: 0 (need at least 3)</i>"));

    m_statusLabel->setText(tr("Click on UPPER JAW (beige) to place first landmark"));
}

void MainWindow::onLandmarkPicked(const Landmark& landmark)
{
    int pairCount = m_glWidget->landmarkPairCount();

    if (landmark.isMaxilla) {
        // Just picked maxilla, now need mandible
        m_landmarkInstructionLabel->setText(
            tr("<b>Good!</b> Now click the <span style='color:#1565C0'>MATCHING POINT</span> on the <span style='color:#1565C0'>LOWER JAW</span> (blue)\n\n"
               "<i>Pairs picked: %1 (need at least 3)</i>").arg(pairCount));
        m_statusLabel->setText(tr("Click matching point on LOWER JAW (blue)"));
    } else {
        // Picked mandible - pair complete, next is maxilla
        m_landmarkInstructionLabel->setText(
            tr("<b>Pair %1 complete!</b>\n\n"
               "Click on <span style='color:#E65100'>UPPER JAW</span> (beige) for next point.\n\n"
               "<i>Pairs picked: %2 (need at least 3)</i>").arg(pairCount).arg(pairCount));
        m_statusLabel->setText(tr("Click on UPPER JAW (beige) for next landmark"));
    }
}

void MainWindow::onLandmarkPairComplete(int pairIndex)
{
    int pairCount = pairIndex + 1;

    // Enable apply button when we have at least 3 pairs
    if (pairCount >= 3) {
        m_landmarkApplyBtn->setEnabled(true);
        m_landmarkInstructionLabel->setText(
            tr("<b>Ready!</b> You have %1 landmark pairs.\n\n"
               "You can add more pairs for better accuracy,\n"
               "or click <b>Apply Alignment</b> to proceed.\n\n"
               "<i>Tip: 3-5 pairs across different teeth works best</i>").arg(pairCount));
    }
}

void MainWindow::applyLandmarkAlignment()
{
    const auto& landmarks = m_glWidget->landmarks();
    int pairCount = m_glWidget->landmarkPairCount();

    if (pairCount < 3) {
        QMessageBox::warning(this, tr("Error"), tr("Need at least 3 landmark pairs."));
        return;
    }

    // Gather corresponding points
    std::vector<Eigen::Vector3f> maxillaPoints, mandiblePoints;

    for (int i = 0; i < pairCount; ++i) {
        for (const auto& lm : landmarks) {
            if (lm.index == i) {
                if (lm.isMaxilla) {
                    maxillaPoints.push_back(lm.position);
                } else {
                    mandiblePoints.push_back(lm.position);
                }
            }
        }
    }

    if (maxillaPoints.size() != mandiblePoints.size()) {
        QMessageBox::warning(this, tr("Error"), tr("Mismatched landmark pairs."));
        return;
    }

    // Apply landmark alignment via BiteSimulator
    m_statusLabel->setText(tr("Applying landmark alignment..."));
    QApplication::processEvents();

    m_biteSimulator->alignFromLandmarks(maxillaPoints, mandiblePoints);

    // Update view
    m_glWidget->updateMandibleFromSimulator(m_biteSimulator->mandible());
    updateContactPointsVisualization();
    updateMetricsDisplay();

    // Exit landmark mode
    cancelLandmarkPicking();

    m_statusLabel->setText(tr("Landmark alignment applied! Run Quick Bite or Optimize for fine-tuning."));
}

void MainWindow::cancelLandmarkPicking()
{
    // Disable landmark picking mode
    m_glWidget->setLandmarkPickingMode(false);
    m_glWidget->clearLandmarks();

    // Restore UI
    m_landmarkWidget->hide();
    m_landmarkBtn->show();
    m_roughAlignBtn->setEnabled(true);
    m_quickBiteBtn->setEnabled(true);
    m_optimizeBtn->setEnabled(true);
    m_landmarkApplyBtn->setEnabled(false);

    m_statusLabel->setText(tr("Landmark picking cancelled"));
}

// ========== Manual Jaw Movement ==========

void MainWindow::onJawMoved()
{
    // Lightweight update - skip expensive calculations
    // User clicks "Calculate Score" button when ready
    QString jawName = m_glWidget->isMovingMaxilla() ? "Upper" : "Lower";
    m_statusLabel->setText(tr("[%1] Moved - click 'Calculate Score' to update metrics").arg(jawName));
}

void MainWindow::onJawSelectionChanged(bool movingMaxilla)
{
    // Update the checkbox to match current selection
    m_moveMaxillaCheck->blockSignals(true);
    m_moveMaxillaCheck->setChecked(movingMaxilla);
    m_moveMaxillaCheck->blockSignals(false);

    QString jawName = movingMaxilla ? "Upper Jaw (Maxilla)" : "Lower Jaw (Mandible)";
    m_statusLabel->setText(tr("Selected: %1").arg(jawName));
}

void MainWindow::onJawMovedFast()
{
    // Lightweight update during drag - skip expensive calculations
    QString jawName = m_glWidget->isMovingMaxilla() ? "Upper" : "Lower";
    m_statusLabel->setText(tr("[%1] Moving...").arg(jawName));
}

void MainWindow::applyManualChanges()
{
    // Calculate metrics when user clicks Apply button
    if (!m_biteSimulator->maxilla() || !m_biteSimulator->mandible()) return;

    m_statusLabel->setText(tr("Calculating..."));
    QApplication::processEvents();

    updateContactPointsVisualization();
    updateMetricsDisplay();

    m_statusLabel->setText(tr("Score updated"));
}

void MainWindow::onSliderMoved()
{
}

void MainWindow::moveJawByStep(int axis, float amount)
{
    Q_UNUSED(axis);
    Q_UNUSED(amount);
}

void MainWindow::runCEMOptimization()
{
    if (!m_biteSimulator->maxilla() || !m_biteSimulator->mandible()) {
        QMessageBox::warning(this, tr("No Data"), tr("Load bite data first."));
        return;
    }

    m_initialReward = m_biteSimulator->computeFastReward();
    m_optBestReward = m_initialReward;

    m_optimizationRunning = true;
    m_optimizeBtn->setEnabled(false);
    m_cemBtn->setEnabled(false);
    m_stopBtn->show();
    m_biteProgressBar->show();
    m_biteProgressBar->setRange(0, 100);
    m_biteProgressBar->setValue(0);
    m_statusLabel->setText(tr("CEM Optimizing..."));
    m_biteSimulator->cacheSamplePoints();

    m_optStep = 0;
    auto future = QtConcurrent::run([this]() {
        for (int i = 0; i < 30 && m_optimizationRunning; ++i) {
            double reward = m_biteSimulator->runAdamStep(0.08f);
            if (reward > m_optBestReward) m_optBestReward = reward;
            m_optStep = i + 1;
        }
        return m_optBestReward;
    });

    if (!m_optimizationWatcher) {
        m_optimizationWatcher = new QFutureWatcher<double>(this);
        connect(m_optimizationWatcher, &QFutureWatcher<double>::finished, this, [this]() {
            m_glWidget->updateMandibleFromSimulator(m_biteSimulator->mandible());
            m_glWidget->update();
            onOptimizationFinished();
        });
    }
    m_optimizationWatcher->setFuture(future);

    m_optTimer = new QTimer(this);
    connect(m_optTimer, &QTimer::timeout, this, [this]() {
        if (!m_optimizationRunning) {
            m_optTimer->stop();
            m_optTimer->deleteLater();
            m_optTimer = nullptr;
            return;
        }
        m_biteProgressBar->setValue(m_optStep * 100 / 30);
        m_statusLabel->setText(tr("CEM [%1/30] Score: %2%").arg(m_optStep).arg((m_optBestReward + 1.0) / 2.0 * 100.0, 0, 'f', 1));
    });
    m_optTimer->start(50);
}

void MainWindow::runESOptimization()
{
}

void MainWindow::savePolicyNetwork()
{
    if (!m_esOptimizer || !m_esOptimizer->policy()) {
        QMessageBox::warning(this, tr("No Policy"), tr("Train ES first."));
        return;
    }

    QString path = QFileDialog::getSaveFileName(this, tr("Save Policy"), "", tr("Policy Files (*.policy)"));
    if (path.isEmpty()) return;

    if (m_esOptimizer->policy()->save(path.toStdString())) {
        m_statusLabel->setText(tr("Policy saved to %1").arg(path));
    } else {
        QMessageBox::warning(this, tr("Error"), tr("Failed to save policy."));
    }
}

void MainWindow::loadPolicyNetwork()
{
    QString path = QFileDialog::getOpenFileName(this, tr("Load Policy"), "", tr("Policy Files (*.policy)"));
    if (path.isEmpty()) return;

    if (!m_esOptimizer) {
        ESOptimizer::Config config;
        m_esOptimizer = std::make_unique<ESOptimizer>(m_biteSimulator.get(), config);
    }

    if (m_esOptimizer->policy()->load(path.toStdString())) {
        m_statusLabel->setText(tr("Policy loaded from %1").arg(path));
    } else {
        QMessageBox::warning(this, tr("Error"), tr("Failed to load policy."));
    }
}

void MainWindow::runPPOOptimization()
{
}
