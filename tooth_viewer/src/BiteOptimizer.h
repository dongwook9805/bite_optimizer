#ifndef BITEOPTIMIZER_H
#define BITEOPTIMIZER_H

#include <QString>
#include <QObject>
#include <QProcess>
#include <QJsonObject>

class BiteOptimizer : public QObject
{
    Q_OBJECT

public:
    struct OptimizationResult {
        bool success = false;
        double initialReward = 0.0;
        double finalReward = 0.0;
        double improvement = 0.0;
        QString outputPath;
        QString errorMessage;
        QJsonObject metrics;
    };

    explicit BiteOptimizer(QObject* parent = nullptr);
    ~BiteOptimizer();

    // Run optimization asynchronously
    void optimizeAsync(const QString& maxillaPath, const QString& mandiblePath,
                       const QString& outputPath, bool icpOnly = false, int maxSteps = 100);

    // Get last result
    OptimizationResult lastResult() const { return m_lastResult; }

signals:
    void optimizationFinished(bool success);
    void optimizationProgress(const QString& message);

private slots:
    void onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void onProcessOutput();
    void onProcessError();

private:
    QString findPythonExecutable();
    QString findOptimizeScript();
    OptimizationResult parseOutput(const QString& output);

    QProcess* m_process = nullptr;
    OptimizationResult m_lastResult;
    QString m_outputBuffer;
};

#endif // BITEOPTIMIZER_H
