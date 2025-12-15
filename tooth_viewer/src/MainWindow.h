#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QProgressBar>
#include <QDockWidget>
#include <QCheckBox>
#include <QGroupBox>
#include <QVBoxLayout>
#include <vector>
#include "GLWidget.h"
#include "Segmentation.h"

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

    // Layer visibility
    void onMeshVisibilityChanged(bool visible);
    void onPointCloudVisibilityChanged(bool visible);
    void onLabelVisibilityChanged(int label, bool visible);

private:
    void setupUI();
    void setupMenuBar();
    void setupToolBar();
    void setupStatusBar();
    void setupSidePanel();
    void updateLabelCheckboxes();
    void createLabelCheckboxes();

    GLWidget* m_glWidget;
    QLabel* m_statusLabel;
    QProgressBar* m_progressBar;
    QString m_currentFilePath;
    QString m_segmentedFilePath;

    Segmentation* m_segmentation;
    bool m_isUpperJaw = true;

    // Side panel
    QDockWidget* m_sidePanel;
    QGroupBox* m_meshGroup;
    QGroupBox* m_segmentationGroup;
    QLabel* m_emptyLabel;
    QWidget* m_labelsWidget;
    QVBoxLayout* m_labelsLayout;
    QCheckBox* m_meshVisibleCheck;
    QCheckBox* m_pointCloudVisibleCheck;
    std::vector<QCheckBox*> m_labelCheckboxes;  // 0-16 for gingiva + 16 teeth
};

#endif // MAINWINDOW_H
