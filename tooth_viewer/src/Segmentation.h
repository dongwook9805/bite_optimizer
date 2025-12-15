#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <QString>
#include <QObject>
#include <QProcess>
#include <functional>

class Segmentation : public QObject
{
    Q_OBJECT

public:
    explicit Segmentation(QObject* parent = nullptr);
    ~Segmentation();

    // Run segmentation on input file, save result to output file
    bool runSegmentation(const QString& inputPath, const QString& outputPath, bool isUpper = true);

    // Async version
    void runSegmentationAsync(const QString& inputPath, const QString& outputPath, bool isUpper = true);

    QString lastError() const { return m_lastError; }

signals:
    void segmentationFinished(bool success, const QString& outputPath);
    void segmentationProgress(const QString& message);

private slots:
    void onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void onProcessOutput();
    void onProcessError();

private:
    QString findPythonExecutable();
    QString findSegmentScript();

    QProcess* m_process = nullptr;
    QString m_lastError;
    QString m_outputPath;
};

#endif // SEGMENTATION_H
