#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include "GLWidget.h"

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

private:
    void setupUI();
    void setupMenuBar();
    void setupToolBar();
    void setupStatusBar();

    GLWidget* m_glWidget;
    QLabel* m_statusLabel;
    QString m_currentFilePath;
};

#endif // MAINWINDOW_H
