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
#include <QApplication>
#include <QFrame>

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

    m_optimizationTimer = new QTimer(this);
    connect(m_optimizationTimer, &QTimer::timeout, this, &MainWindow::onOptimizationStep);

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

    // ===== Bite Optimization Group (Main UX) =====
    m_biteGroup = new QGroupBox(tr("Bite Optimization"));
    QVBoxLayout* biteLayout = new QVBoxLayout(m_biteGroup);
    biteLayout->setSpacing(8);

    // Visibility toggles
    m_maxillaVisibleCheck = new QCheckBox(tr("Upper Jaw (Maxilla)"));
    m_maxillaVisibleCheck->setChecked(true);
    connect(m_maxillaVisibleCheck, &QCheckBox::toggled, this, &MainWindow::onMaxillaVisibilityChanged);
    biteLayout->addWidget(m_maxillaVisibleCheck);

    m_mandibleVisibleCheck = new QCheckBox(tr("Lower Jaw (Mandible)"));
    m_mandibleVisibleCheck->setChecked(true);
    connect(m_mandibleVisibleCheck, &QCheckBox::toggled, this, &MainWindow::onMandibleVisibilityChanged);
    biteLayout->addWidget(m_mandibleVisibleCheck);

    m_contactPointsCheck = new QCheckBox(tr("Contact Coloring"));
    m_contactPointsCheck->setChecked(true);
    m_contactPointsCheck->setToolTip(tr("Colors teeth by contact: Green = Good, Red = Penetration, Blue = Far"));
    connect(m_contactPointsCheck, &QCheckBox::toggled, this, &MainWindow::onContactPointsVisibilityChanged);
    biteLayout->addWidget(m_contactPointsCheck);

    // Separator
    QFrame* line1 = new QFrame();
    line1->setFrameShape(QFrame::HLine);
    line1->setFrameShadow(QFrame::Sunken);
    biteLayout->addWidget(line1);

    // Rough Align button
    m_roughAlignBtn = new QPushButton(tr("Rough Align"));
    m_roughAlignBtn->setToolTip(tr("Position mandible below maxilla (use if meshes are overlapping)"));
    m_roughAlignBtn->setStyleSheet("QPushButton { padding: 6px; }");
    connect(m_roughAlignBtn, &QPushButton::clicked, this, &MainWindow::roughAlignJaws);
    biteLayout->addWidget(m_roughAlignBtn);

    // Landmark-based alignment button
    m_landmarkBtn = new QPushButton(tr("🎯 Pick Landmarks"));
    m_landmarkBtn->setToolTip(tr("Click matching points on upper and lower jaw for precise alignment"));
    m_landmarkBtn->setStyleSheet("QPushButton { padding: 6px; background-color: #2196F3; color: white; }");
    connect(m_landmarkBtn, &QPushButton::clicked, this, &MainWindow::startLandmarkPicking);
    biteLayout->addWidget(m_landmarkBtn);

    // Landmark picking widget (initially hidden)
    m_landmarkWidget = new QWidget();
    QVBoxLayout* landmarkLayout = new QVBoxLayout(m_landmarkWidget);
    landmarkLayout->setContentsMargins(0, 0, 0, 0);
    landmarkLayout->setSpacing(4);

    m_landmarkInstructionLabel = new QLabel();
    m_landmarkInstructionLabel->setWordWrap(true);
    m_landmarkInstructionLabel->setStyleSheet("QLabel { background-color: #E3F2FD; padding: 8px; border-radius: 4px; }");
    landmarkLayout->addWidget(m_landmarkInstructionLabel);

    QHBoxLayout* landmarkBtnLayout = new QHBoxLayout();
    m_landmarkApplyBtn = new QPushButton(tr("Apply Alignment"));
    m_landmarkApplyBtn->setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }");
    m_landmarkApplyBtn->setEnabled(false);
    connect(m_landmarkApplyBtn, &QPushButton::clicked, this, &MainWindow::applyLandmarkAlignment);
    landmarkBtnLayout->addWidget(m_landmarkApplyBtn);

    m_landmarkCancelBtn = new QPushButton(tr("Cancel"));
    connect(m_landmarkCancelBtn, &QPushButton::clicked, this, &MainWindow::cancelLandmarkPicking);
    landmarkBtnLayout->addWidget(m_landmarkCancelBtn);

    landmarkLayout->addLayout(landmarkBtnLayout);
    m_landmarkWidget->hide();
    biteLayout->addWidget(m_landmarkWidget);

    // Step 2: Quick Bite (ICP)
    m_quickBiteBtn = new QPushButton(tr("Quick Bite (Preview)"));
    m_quickBiteBtn->setToolTip(tr("Fast initial alignment - gets you close"));
    m_quickBiteBtn->setStyleSheet("QPushButton { padding: 8px; font-weight: bold; }");
    connect(m_quickBiteBtn, &QPushButton::clicked, this, &MainWindow::runQuickBite);
    biteLayout->addWidget(m_quickBiteBtn);

    // Step 3: Optimize Bite
    m_optimizeBtn = new QPushButton(tr("Find Best Occlusion"));
    m_optimizeBtn->setToolTip(tr("Fine-tune the bite for optimal contact"));
    m_optimizeBtn->setStyleSheet("QPushButton { padding: 8px; font-weight: bold; background-color: #4CAF50; color: white; }");
    connect(m_optimizeBtn, &QPushButton::clicked, this, &MainWindow::runOptimizeBite);
    biteLayout->addWidget(m_optimizeBtn);

    // Stop button (initially hidden)
    m_stopBtn = new QPushButton(tr("Stop & Keep Current"));
    m_stopBtn->setStyleSheet("QPushButton { padding: 6px; background-color: #f44336; color: white; }");
    m_stopBtn->hide();
    connect(m_stopBtn, &QPushButton::clicked, this, &MainWindow::stopOptimization);
    biteLayout->addWidget(m_stopBtn);

    // Progress bar for optimization
    m_biteProgressBar = new QProgressBar();
    m_biteProgressBar->setTextVisible(true);
    m_biteProgressBar->hide();
    biteLayout->addWidget(m_biteProgressBar);

    // Phase indicators
    QFrame* phaseFrame = new QFrame();
    phaseFrame->setFrameShape(QFrame::StyledPanel);
    QVBoxLayout* phaseLayout = new QVBoxLayout(phaseFrame);
    phaseLayout->setSpacing(4);
    phaseLayout->setContentsMargins(8, 8, 8, 8);

    m_phaseLabel = new QLabel(tr("Optimization Phases:"));
    m_phaseLabel->setStyleSheet("font-weight: bold;");
    phaseLayout->addWidget(m_phaseLabel);

    m_phase1Label = new QLabel(tr("  Phase 1: Stabilizing contact"));
    m_phase2Label = new QLabel(tr("  Phase 2: Balancing occlusion"));
    m_phase3Label = new QLabel(tr("  Phase 3: Fine adjustment"));

    QString inactiveStyle = "color: gray;";
    m_phase1Label->setStyleSheet(inactiveStyle);
    m_phase2Label->setStyleSheet(inactiveStyle);
    m_phase3Label->setStyleSheet(inactiveStyle);

    phaseLayout->addWidget(m_phase1Label);
    phaseLayout->addWidget(m_phase2Label);
    phaseLayout->addWidget(m_phase3Label);
    phaseFrame->hide();

    biteLayout->addWidget(phaseFrame);

    // Separator
    QFrame* line2 = new QFrame();
    line2->setFrameShape(QFrame::HLine);
    line2->setFrameShadow(QFrame::Sunken);
    biteLayout->addWidget(line2);

    // Reset and Export
    QHBoxLayout* actionLayout = new QHBoxLayout();

    m_resetBtn = new QPushButton(tr("Reset"));
    m_resetBtn->setToolTip(tr("Return to initial position"));
    connect(m_resetBtn, &QPushButton::clicked, this, &MainWindow::resetBiteAlignment);
    actionLayout->addWidget(m_resetBtn);

    m_exportBtn = new QPushButton(tr("Export"));
    m_exportBtn->setToolTip(tr("Save aligned meshes"));
    connect(m_exportBtn, &QPushButton::clicked, this, &MainWindow::exportAlignedMeshes);
    actionLayout->addWidget(m_exportBtn);

    biteLayout->addLayout(actionLayout);

    // Before/After toggle
    m_beforeAfterCheck = new QCheckBox(tr("Show Before State"));
    m_beforeAfterCheck->setToolTip(tr("Toggle to compare before and after alignment"));
    connect(m_beforeAfterCheck, &QCheckBox::toggled, this, &MainWindow::toggleBeforeAfter);
    biteLayout->addWidget(m_beforeAfterCheck);

    // Separator
    QFrame* line3 = new QFrame();
    line3->setFrameShape(QFrame::HLine);
    line3->setFrameShadow(QFrame::Sunken);
    biteLayout->addWidget(line3);

    // ===== Manual Movement Controls =====
    m_movementGroup = new QGroupBox(tr("Manual Adjustment"));
    QVBoxLayout* moveLayout = new QVBoxLayout(m_movementGroup);
    moveLayout->setSpacing(8);

    // Which jaw to move
    m_moveMaxillaCheck = new QCheckBox(tr("Move Upper Jaw (Maxilla)"));
    m_moveMaxillaCheck->setChecked(false);  // Default: move lower jaw
    m_moveMaxillaCheck->setStyleSheet("font-weight: bold;");
    connect(m_moveMaxillaCheck, &QCheckBox::toggled, this, [this](bool checked) {
        m_glWidget->setMovingMaxilla(checked);
    });
    moveLayout->addWidget(m_moveMaxillaCheck);

    // Mouse control instructions (Exocad style)
    QLabel* instructionLabel = new QLabel(
        tr("<b>View Controls:</b><br>"
           "Right-drag = Rotate view<br>"
           "Left+Right drag = Pan view<br>"
           "Wheel = Zoom<br>"
           "Wheel-click = Set pivot<br>"
           "<br>"
           "<b>Mesh Controls:</b><br>"
           "Left-drag = Move jaw<br>"
           "Ctrl+drag = Rotate jaw"));
    instructionLabel->setStyleSheet("QLabel { background-color: #f0f0f0; padding: 10px; border-radius: 4px; }");
    instructionLabel->setWordWrap(true);
    moveLayout->addWidget(instructionLabel);

    // Initialize slider pointers to nullptr (not used but needed for header compatibility)
    m_sliderX = nullptr;
    m_sliderY = nullptr;
    m_sliderZ = nullptr;
    m_sliderRotX = nullptr;
    m_sliderRotY = nullptr;
    m_sliderRotZ = nullptr;
    m_labelX = nullptr;
    m_labelY = nullptr;
    m_labelZ = nullptr;
    m_labelRotX = nullptr;
    m_labelRotY = nullptr;
    m_labelRotZ = nullptr;

    biteLayout->addWidget(m_movementGroup);

    // Hidden keyboard help (keep for reference)
    m_keyboardHelpLabel = new QLabel();
    m_keyboardHelpLabel->hide();

    // Apply button for manual changes
    m_applyManualBtn = new QPushButton(tr("Calculate Score"));
    m_applyManualBtn->setToolTip(tr("Calculate occlusion metrics for current position"));
    m_applyManualBtn->setStyleSheet("QPushButton { padding: 10px; background-color: #2196F3; color: white; font-weight: bold; font-size: 14px; }");
    connect(m_applyManualBtn, &QPushButton::clicked, this, &MainWindow::applyManualChanges);
    biteLayout->addWidget(m_applyManualBtn);

    // Separator
    QFrame* line4 = new QFrame();
    line4->setFrameShape(QFrame::HLine);
    line4->setFrameShadow(QFrame::Sunken);
    biteLayout->addWidget(line4);

    // Metrics display (user-friendly format)
    m_metricsLabel = new QLabel();
    m_metricsLabel->setWordWrap(true);
    m_metricsLabel->setStyleSheet("QLabel { background-color: #f5f5f5; padding: 8px; border-radius: 4px; }");
    biteLayout->addWidget(m_metricsLabel);

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

    // AI menu
    QMenu* aiMenu = menuBar()->addMenu(tr("&AI"));

    QAction* segmentAction = aiMenu->addAction(tr("&Tooth Segmentation"));
    segmentAction->setShortcut(Qt::CTRL | Qt::Key_T);
    connect(segmentAction, &QAction::triggered, this, &MainWindow::runAISegmentation);

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

    // Store initial state for comparison
    OrthodonticMetrics metrics = m_biteSimulator->computeMetrics();
    m_initialReward = m_biteSimulator->computeReward(metrics);
    m_previousReward = m_initialReward;

    m_optimizationSteps = 0;
    m_currentPhase = 1;

    // UI state for optimization
    m_quickBiteBtn->setEnabled(false);
    m_optimizeBtn->hide();
    m_stopBtn->show();
    m_biteProgressBar->show();
    m_biteProgressBar->setRange(0, m_maxOptimizationSteps);
    m_biteProgressBar->setValue(0);

    // Show phase indicators
    m_phaseLabel->parentWidget()->show();
    setOptimizationPhase(1);

    m_statusLabel->setText(tr("Optimizing..."));

    m_optimizationTimer->start(30);  // 30ms interval for smoother animation
}

void MainWindow::stopOptimization()
{
    m_optimizationTimer->stop();

    // Restore UI
    m_quickBiteBtn->setEnabled(true);
    m_optimizeBtn->show();
    m_stopBtn->hide();
    m_biteProgressBar->hide();
    m_currentPhase = 0;

    m_statusLabel->setText(tr("Optimization stopped - current state preserved"));
    updateMetricsDisplay();
}

void MainWindow::onOptimizationStep()
{
    if (m_optimizationSteps >= m_maxOptimizationSteps) {
        stopOptimization();
        m_statusLabel->setText(tr("Optimization complete!"));
        return;
    }

    // Determine phase based on progress
    int newPhase = 1;
    if (m_optimizationSteps > m_maxOptimizationSteps * 0.33) newPhase = 2;
    if (m_optimizationSteps > m_maxOptimizationSteps * 0.66) newPhase = 3;

    if (newPhase != m_currentPhase) {
        m_currentPhase = newPhase;
        setOptimizationPhase(newPhase);
    }

    // Run optimization step
    m_biteSimulator->optimizeStep(0.01);

    // Update visualization every 3 steps for smooth animation
    if (m_optimizationSteps % 3 == 0) {
        m_glWidget->updateMandibleFromSimulator(m_biteSimulator->mandible());
        updateContactPointsVisualization();
        updateMetricsDisplay();
    }

    m_optimizationSteps++;
    m_biteProgressBar->setValue(m_optimizationSteps);
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
        m_metricsLabel->setText(tr("Load bite data to see metrics"));
        return;
    }

    OrthodonticMetrics metrics = m_biteSimulator->computeMetrics();
    double reward = m_biteSimulator->computeReward(metrics);

    // === 제1원칙: 상호 보호 교합 ===
    QString protectionStatus;
    QString protectionColor;
    if (metrics.protection_ratio >= 0.90) {
        protectionStatus = "Excellent";
        protectionColor = "#4CAF50";  // Green
    } else if (metrics.protection_ratio >= 0.70) {
        protectionStatus = "Good";
        protectionColor = "#FFC107";  // Yellow
    } else {
        protectionStatus = "Poor";
        protectionColor = "#F44336";  // Red
    }

    // === 제2원칙: 좌우 균형 ===
    QString balanceStatus;
    QString balanceColor;
    if (metrics.balance_error < 0.15) {
        balanceStatus = "Balanced";
        balanceColor = "#4CAF50";
    } else if (metrics.balance_error < 0.30) {
        balanceStatus = "Slight Imbalance";
        balanceColor = "#FFC107";
    } else {
        balanceStatus = "Uneven";
        balanceColor = "#F44336";
    }

    // === 제3원칙: 치축 방향 ===
    QString axialStatus;
    QString axialColor;
    if (metrics.axial_alignment_score >= 0.8 && metrics.lateral_force_penalty < 0.2) {
        axialStatus = "Vertical";
        axialColor = "#4CAF50";
    } else if (metrics.lateral_force_penalty < 0.4) {
        axialStatus = "Acceptable";
        axialColor = "#FFC107";
    } else {
        axialStatus = "Lateral Force!";
        axialColor = "#F44336";
    }

    // Score change indicator
    QString scoreChange;
    double delta = reward - m_previousReward;
    if (std::abs(delta) > 0.001) {
        scoreChange = QString(" (%1%2)").arg(delta > 0 ? "+" : "").arg(delta, 0, 'f', 3);
    }
    m_previousReward = reward;

    // Convert score to percentage (0-100)
    double scorePercent = (reward + 1.0) / 2.0 * 100.0;

    QString text = QString(
        "<b style='font-size:14px;'>Occlusion Score: %1%%2</b><br><br>"
        "<b>1. Molar Protection:</b> <span style='color:%3'>%4</span><br>"
        "   Post/Total: %5%<br><br>"
        "<b>2. L/R Balance:</b> <span style='color:%6'>%7</span><br>"
        "   L:%8 R:%9<br><br>"
        "<b>3. Axial Load:</b> <span style='color:%10'>%11</span><br>"
        "   Vertical: %12%<br><br>"
        "<b>4. Distribution:</b> %13%<br><br>"
        "<small>Contacts: %14 | Penetration: %15</small>")
        .arg(scorePercent, 0, 'f', 1)
        .arg(scoreChange)
        .arg(protectionColor)
        .arg(protectionStatus)
        .arg(metrics.protection_ratio * 100, 0, 'f', 0)
        .arg(balanceColor)
        .arg(balanceStatus)
        .arg(metrics.force_left, 0, 'f', 1)
        .arg(metrics.force_right, 0, 'f', 1)
        .arg(axialColor)
        .arg(axialStatus)
        .arg(metrics.axial_alignment_score * 100, 0, 'f', 0)
        .arg(metrics.contact_evenness * 100, 0, 'f', 0)
        .arg(metrics.contact_point_count)
        .arg(static_cast<int>(metrics.penetration_count));

    m_metricsLabel->setText(text);
}

void MainWindow::updatePhaseDisplay()
{
    // Called when phase changes during optimization
    setOptimizationPhase(m_currentPhase);
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
    if (!m_biteSimulator->maxilla() || !m_biteSimulator->mandible()) return;

    // Get translation slider values (0.1mm units)
    float dx = m_sliderX->value() * 0.1f;
    float dy = m_sliderY->value() * 0.1f;
    float dz = m_sliderZ->value() * 0.1f;

    // Get rotation slider values (0.1° units)
    float rx = m_sliderRotX->value() * 0.1f;
    float ry = m_sliderRotY->value() * 0.1f;
    float rz = m_sliderRotZ->value() * 0.1f;

    // Update translation labels
    m_labelX->setText(QString::number(dx, 'f', 1));
    m_labelY->setText(QString::number(dy, 'f', 1));
    m_labelZ->setText(QString::number(dz, 'f', 1));

    // Update rotation labels
    m_labelRotX->setText(QString::number(rx, 'f', 1));
    m_labelRotY->setText(QString::number(ry, 'f', 1));
    m_labelRotZ->setText(QString::number(rz, 'f', 1));

    // Track previous values to compute deltas
    static float lastX = 0, lastY = 0, lastZ = 0;
    static float lastRX = 0, lastRY = 0, lastRZ = 0;

    float deltaX = dx - lastX;
    float deltaY = dy - lastY;
    float deltaZ = dz - lastZ;
    float deltaRX = rx - lastRX;
    float deltaRY = ry - lastRY;
    float deltaRZ = rz - lastRZ;

    lastX = dx; lastY = dy; lastZ = dz;
    lastRX = rx; lastRY = ry; lastRZ = rz;

    bool hasTranslation = std::abs(deltaX) > 0.001f || std::abs(deltaY) > 0.001f || std::abs(deltaZ) > 0.001f;
    bool hasRotation = std::abs(deltaRX) > 0.001f || std::abs(deltaRY) > 0.001f || std::abs(deltaRZ) > 0.001f;

    if (hasTranslation || hasRotation) {
        Eigen::Vector3f translation(deltaX, deltaY, deltaZ);
        Eigen::Vector3f rotation(deltaRX, deltaRY, deltaRZ);
        m_biteSimulator->applyTransform(rotation, translation, m_moveMaxillaCheck->isChecked());

        // Update visualization
        if (m_moveMaxillaCheck->isChecked()) {
            m_glWidget->updateMaxillaFromSimulator(m_biteSimulator->maxilla());
        } else {
            m_glWidget->updateMandibleFromSimulator(m_biteSimulator->mandible());
        }

        m_statusLabel->setText(tr("Pos: %.1f, %.1f, %.1f mm | Rot: %.1f, %.1f, %.1f deg")
            .arg(dx).arg(dy).arg(dz).arg(rx).arg(ry).arg(rz));
    }
}

void MainWindow::moveJawByStep(int axis, float amount)
{
    if (!m_biteSimulator->maxilla() || !m_biteSimulator->mandible()) return;

    // Update the appropriate slider
    QSlider* slider = nullptr;
    bool isRotation = false;

    switch (axis) {
        // Translation (mm)
        case 0: slider = m_sliderX; break;
        case 1: slider = m_sliderY; break;
        case 2: slider = m_sliderZ; break;
        // Rotation (degrees)
        case 3: slider = m_sliderRotX; isRotation = true; break;
        case 4: slider = m_sliderRotY; isRotation = true; break;
        case 5: slider = m_sliderRotZ; isRotation = true; break;
        default: return;
    }

    // Convert amount to slider units
    // Translation: 0.1mm per unit, Rotation: 0.1° per unit
    int delta = static_cast<int>(amount * 10);
    int newValue = slider->value() + delta;
    newValue = qBound(slider->minimum(), newValue, slider->maximum());
    slider->setValue(newValue);

    // onSliderMoved will be called automatically
}
