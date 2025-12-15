#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QProgressBar>
#include <QDockWidget>
#include <QCheckBox>
#include <QGroupBox>
#include <QVBoxLayout>
#include <QTimer>
#include <QPushButton>
#include <QSlider>
#include <QDoubleSpinBox>
#include <vector>
#include <memory>
#include "GLWidget.h"
#include "Segmentation.h"
#include "BiteSimulator.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

private slots:
    void openFile();
    void saveFile();
    void toggleWireframe(bool checked);
    void resetView();
    void onMeshLoaded(size_t vertices, size_t faces);

    // AI Segmentation (CrossTooth)
    void runAISegmentation();
    void onSegmentationFinished(bool success, const QString& outputPath);
    void onSegmentationProgress(const QString& message);

    // Bite Optimization
    void loadBiteData();
    void roughAlignJaws();    // Rough positioning of jaws
    void runQuickBite();      // Step 2: Quick rough alignment
    void runOptimizeBite();   // Step 3: RL optimization
    void stopOptimization();  // Stop button
    void resetBiteAlignment();
    void onBiteDataLoaded();
    void onOptimizationStep();
    void toggleBeforeAfter(bool showAfter);  // Before/After toggle
    void exportAlignedMeshes();

    // Landmark alignment
    void startLandmarkPicking();
    void onLandmarkPicked(const Landmark& landmark);
    void onLandmarkPairComplete(int pairIndex);
    void applyLandmarkAlignment();
    void cancelLandmarkPicking();

    // Manual jaw movement
    void onJawMoved();  // Called when user moves jaw (lightweight update)
    void onJawMovedFast();  // Called during dragging (skip expensive calculations)
    void onJawSelectionChanged(bool movingMaxilla);  // Called when Tab switches jaw
    void applyManualChanges();  // Apply button - calculate metrics

    // Slider-based movement
    void onSliderMoved();  // Called when any slider changes
    void moveJawByStep(int axis, float amount);  // axis: 0=X, 1=Y, 2=Z, 3=rotX, 4=rotY, 5=rotZ

    // Layer visibility
    void onMeshVisibilityChanged(bool visible);
    void onPointCloudVisibilityChanged(bool visible);
    void onLabelVisibilityChanged(int label, bool visible);
    void onMaxillaVisibilityChanged(bool visible);
    void onMandibleVisibilityChanged(bool visible);
    void onContactPointsVisibilityChanged(bool visible);
    void updateContactPointsVisualization();

private:
    void setupUI();
    void setupMenuBar();
    void setupToolBar();
    void setupStatusBar();
    void setupSidePanel();
    void updateLabelCheckboxes();
    void createLabelCheckboxes();
    void updateMetricsDisplay();
    void updatePhaseDisplay();
    void setOptimizationPhase(int phase);

    GLWidget* m_glWidget;
    QLabel* m_statusLabel;
    QProgressBar* m_progressBar;
    QString m_currentFilePath;
    QString m_segmentedFilePath;

    Segmentation* m_segmentation;
    bool m_isUpperJaw = true;

    // Bite Optimization
    std::unique_ptr<BiteSimulator> m_biteSimulator;
    QTimer* m_optimizationTimer = nullptr;
    int m_optimizationSteps = 0;
    int m_maxOptimizationSteps = 100;
    int m_currentPhase = 0;  // 0=none, 1=stabilizing, 2=balancing, 3=fine-tuning
    QString m_maxillaPath;
    QString m_mandiblePath;
    double m_previousReward = 0.0;
    double m_initialReward = 0.0;
    bool m_showingBeforeState = false;

    // Side panel
    QDockWidget* m_sidePanel;
    QGroupBox* m_meshGroup;
    QGroupBox* m_segmentationGroup;
    QGroupBox* m_biteGroup;
    QLabel* m_emptyLabel;
    QWidget* m_labelsWidget;
    QVBoxLayout* m_labelsLayout;
    QCheckBox* m_meshVisibleCheck;
    QCheckBox* m_pointCloudVisibleCheck;
    QCheckBox* m_maxillaVisibleCheck;
    QCheckBox* m_mandibleVisibleCheck;
    QCheckBox* m_contactPointsCheck;

    // Bite optimization UI elements
    QPushButton* m_roughAlignBtn;
    QPushButton* m_landmarkBtn;
    QPushButton* m_quickBiteBtn;
    QPushButton* m_optimizeBtn;
    QPushButton* m_stopBtn;
    QPushButton* m_resetBtn;
    QPushButton* m_exportBtn;
    QCheckBox* m_beforeAfterCheck;
    QLabel* m_metricsLabel;
    QLabel* m_phaseLabel;
    QProgressBar* m_biteProgressBar;
    QLabel* m_phase1Label;
    QLabel* m_phase2Label;
    QLabel* m_phase3Label;

    // Landmark UI
    QWidget* m_landmarkWidget;
    QLabel* m_landmarkInstructionLabel;
    QPushButton* m_landmarkApplyBtn;
    QPushButton* m_landmarkCancelBtn;

    // Manual movement UI
    QLabel* m_keyboardHelpLabel;
    QPushButton* m_applyManualBtn;

    // Slider controls for jaw movement
    QGroupBox* m_movementGroup;
    // Translation sliders (±50mm)
    QSlider* m_sliderX;
    QSlider* m_sliderY;
    QSlider* m_sliderZ;
    QLabel* m_labelX;
    QLabel* m_labelY;
    QLabel* m_labelZ;
    // Rotation sliders (±180°)
    QSlider* m_sliderRotX;  // Pitch
    QSlider* m_sliderRotY;  // Yaw
    QSlider* m_sliderRotZ;  // Roll
    QLabel* m_labelRotX;
    QLabel* m_labelRotY;
    QLabel* m_labelRotZ;
    QCheckBox* m_moveMaxillaCheck;

    std::vector<QCheckBox*> m_labelCheckboxes;  // 0-16 for gingiva + 16 teeth
};

#endif // MAINWINDOW_H
