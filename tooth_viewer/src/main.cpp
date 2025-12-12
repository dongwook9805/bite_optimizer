#include <QApplication>
#include <QSurfaceFormat>
#include "MainWindow.h"

int main(int argc, char *argv[])
{
    // Set OpenGL version before creating QApplication
    QSurfaceFormat format;
    format.setVersion(3, 3);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setDepthBufferSize(24);
    format.setSamples(4);
    QSurfaceFormat::setDefaultFormat(format);

    QApplication app(argc, argv);
    app.setApplicationName("ToothViewer");
    app.setApplicationVersion("1.0.0");
    app.setOrganizationName("ToothViewer");

    MainWindow mainWindow;
    mainWindow.show();

    return app.exec();
}
