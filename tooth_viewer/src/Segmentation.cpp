#include "Segmentation.h"
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QDebug>

Segmentation::Segmentation(QObject* parent)
    : QObject(parent)
{
}

Segmentation::~Segmentation()
{
    if (m_process) {
        m_process->kill();
        m_process->deleteLater();
    }
}

QString Segmentation::findPythonExecutable()
{
    // Try venv first, then common Python paths
    QStringList pythonPaths = {
        QDir::homePath() + "/Desktop/bite_optimizer/venv/bin/python3",  // Project venv
        "/usr/bin/python3",
        "/usr/local/bin/python3",
        "/opt/homebrew/bin/python3",
        "python3",
        "python"
    };

    for (const QString& path : pythonPaths) {
        if (QFileInfo::exists(path)) {
            return path;
        }
    }

    return "python3";
}

QString Segmentation::findSegmentScript()
{
    // Look for script relative to app bundle or executable
    QString appDir = QCoreApplication::applicationDirPath();

    QStringList searchPaths = {
        appDir + "/../Resources/scripts/segment.py",  // macOS bundle
        appDir + "/scripts/segment.py",               // Development
        appDir + "/../../../scripts/segment.py",      // Build directory
        QDir::homePath() + "/Desktop/bite_optimizer/tooth_viewer/scripts/segment.py"  // Fallback
    };

    for (const QString& path : searchPaths) {
        QFileInfo fi(path);
        if (fi.exists()) {
            return fi.absoluteFilePath();
        }
    }

    return QString();
}

bool Segmentation::runSegmentation(const QString& inputPath, const QString& outputPath, bool isUpper)
{
    QString python = findPythonExecutable();
    QString script = findSegmentScript();

    if (script.isEmpty()) {
        m_lastError = "Segmentation script not found";
        return false;
    }

    QStringList args;
    args << script;
    args << "--input" << inputPath;
    args << "--output" << outputPath;
    if (isUpper) {
        args << "--upper";
    }

    QProcess process;
    process.start(python, args);

    if (!process.waitForStarted(5000)) {
        m_lastError = "Failed to start Python process";
        return false;
    }

    // Wait up to 5 minutes for completion
    if (!process.waitForFinished(300000)) {
        m_lastError = "Segmentation timeout";
        process.kill();
        return false;
    }

    if (process.exitCode() != 0) {
        m_lastError = QString::fromUtf8(process.readAllStandardError());
        return false;
    }

    return QFileInfo::exists(outputPath);
}

void Segmentation::runSegmentationAsync(const QString& inputPath, const QString& outputPath, bool isUpper)
{
    if (m_process) {
        m_process->kill();
        m_process->deleteLater();
    }

    QString python = findPythonExecutable();
    QString script = findSegmentScript();

    if (script.isEmpty()) {
        m_lastError = "Segmentation script not found";
        emit segmentationFinished(false, QString());
        return;
    }

    m_outputPath = outputPath;

    QStringList args;
    args << script;
    args << "--input" << inputPath;
    args << "--output" << outputPath;
    if (isUpper) {
        args << "--upper";
    }

    m_process = new QProcess(this);
    connect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &Segmentation::onProcessFinished);
    connect(m_process, &QProcess::readyReadStandardOutput, this, &Segmentation::onProcessOutput);
    connect(m_process, &QProcess::readyReadStandardError, this, &Segmentation::onProcessError);

    emit segmentationProgress("Starting segmentation...");
    m_process->start(python, args);
}

void Segmentation::onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    bool success = (exitCode == 0 && exitStatus == QProcess::NormalExit);

    if (!success) {
        m_lastError = QString::fromUtf8(m_process->readAllStandardError());
    }

    emit segmentationFinished(success, success ? m_outputPath : QString());

    m_process->deleteLater();
    m_process = nullptr;
}

void Segmentation::onProcessOutput()
{
    QString output = QString::fromUtf8(m_process->readAllStandardOutput());
    emit segmentationProgress(output.trimmed());
}

void Segmentation::onProcessError()
{
    QString error = QString::fromUtf8(m_process->readAllStandardError());
    qDebug() << "Segmentation error:" << error;
}
