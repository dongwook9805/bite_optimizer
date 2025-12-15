/****************************************************************************
** Meta object code from reading C++ file 'MainWindow.h'
**
** Created by: The Qt Meta Object Compiler version 69 (Qt 6.9.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../src/MainWindow.h"
#include <QtGui/qtextcursor.h>
#include <QtCore/qmetatype.h>

#include <QtCore/qtmochelpers.h>

#include <memory>


#include <QtCore/qxptype_traits.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'MainWindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 69
#error "This file was generated using the moc from 6.9.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

#ifndef Q_CONSTINIT
#define Q_CONSTINIT
#endif

QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
QT_WARNING_DISABLE_GCC("-Wuseless-cast")
namespace {
struct qt_meta_tag_ZN10MainWindowE_t {};
} // unnamed namespace

template <> constexpr inline auto MainWindow::qt_create_metaobjectdata<qt_meta_tag_ZN10MainWindowE_t>()
{
    namespace QMC = QtMocConstants;
    QtMocHelpers::StringRefStorage qt_stringData {
        "MainWindow",
        "openFile",
        "",
        "saveFile",
        "toggleWireframe",
        "checked",
        "resetView",
        "onMeshLoaded",
        "size_t",
        "vertices",
        "faces",
        "runAISegmentation",
        "onSegmentationFinished",
        "success",
        "outputPath",
        "onSegmentationProgress",
        "message",
        "loadBiteData",
        "roughAlignJaws",
        "runQuickBite",
        "runOptimizeBite",
        "stopOptimization",
        "resetBiteAlignment",
        "onBiteDataLoaded",
        "onOptimizationStep",
        "toggleBeforeAfter",
        "showAfter",
        "exportAlignedMeshes",
        "startLandmarkPicking",
        "onLandmarkPicked",
        "Landmark",
        "landmark",
        "onLandmarkPairComplete",
        "pairIndex",
        "applyLandmarkAlignment",
        "cancelLandmarkPicking",
        "onJawMoved",
        "onJawMovedFast",
        "onJawSelectionChanged",
        "movingMaxilla",
        "applyManualChanges",
        "onSliderMoved",
        "moveJawByStep",
        "axis",
        "amount",
        "onMeshVisibilityChanged",
        "visible",
        "onPointCloudVisibilityChanged",
        "onLabelVisibilityChanged",
        "label",
        "onMaxillaVisibilityChanged",
        "onMandibleVisibilityChanged",
        "onContactPointsVisibilityChanged",
        "updateContactPointsVisualization"
    };

    QtMocHelpers::UintData qt_methods {
        // Slot 'openFile'
        QtMocHelpers::SlotData<void()>(1, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'saveFile'
        QtMocHelpers::SlotData<void()>(3, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'toggleWireframe'
        QtMocHelpers::SlotData<void(bool)>(4, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Bool, 5 },
        }}),
        // Slot 'resetView'
        QtMocHelpers::SlotData<void()>(6, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onMeshLoaded'
        QtMocHelpers::SlotData<void(size_t, size_t)>(7, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { 0x80000000 | 8, 9 }, { 0x80000000 | 8, 10 },
        }}),
        // Slot 'runAISegmentation'
        QtMocHelpers::SlotData<void()>(11, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onSegmentationFinished'
        QtMocHelpers::SlotData<void(bool, const QString &)>(12, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Bool, 13 }, { QMetaType::QString, 14 },
        }}),
        // Slot 'onSegmentationProgress'
        QtMocHelpers::SlotData<void(const QString &)>(15, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::QString, 16 },
        }}),
        // Slot 'loadBiteData'
        QtMocHelpers::SlotData<void()>(17, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'roughAlignJaws'
        QtMocHelpers::SlotData<void()>(18, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'runQuickBite'
        QtMocHelpers::SlotData<void()>(19, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'runOptimizeBite'
        QtMocHelpers::SlotData<void()>(20, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'stopOptimization'
        QtMocHelpers::SlotData<void()>(21, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'resetBiteAlignment'
        QtMocHelpers::SlotData<void()>(22, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onBiteDataLoaded'
        QtMocHelpers::SlotData<void()>(23, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onOptimizationStep'
        QtMocHelpers::SlotData<void()>(24, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'toggleBeforeAfter'
        QtMocHelpers::SlotData<void(bool)>(25, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Bool, 26 },
        }}),
        // Slot 'exportAlignedMeshes'
        QtMocHelpers::SlotData<void()>(27, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'startLandmarkPicking'
        QtMocHelpers::SlotData<void()>(28, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onLandmarkPicked'
        QtMocHelpers::SlotData<void(const Landmark &)>(29, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { 0x80000000 | 30, 31 },
        }}),
        // Slot 'onLandmarkPairComplete'
        QtMocHelpers::SlotData<void(int)>(32, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 33 },
        }}),
        // Slot 'applyLandmarkAlignment'
        QtMocHelpers::SlotData<void()>(34, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'cancelLandmarkPicking'
        QtMocHelpers::SlotData<void()>(35, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onJawMoved'
        QtMocHelpers::SlotData<void()>(36, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onJawMovedFast'
        QtMocHelpers::SlotData<void()>(37, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onJawSelectionChanged'
        QtMocHelpers::SlotData<void(bool)>(38, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Bool, 39 },
        }}),
        // Slot 'applyManualChanges'
        QtMocHelpers::SlotData<void()>(40, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onSliderMoved'
        QtMocHelpers::SlotData<void()>(41, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'moveJawByStep'
        QtMocHelpers::SlotData<void(int, float)>(42, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 43 }, { QMetaType::Float, 44 },
        }}),
        // Slot 'onMeshVisibilityChanged'
        QtMocHelpers::SlotData<void(bool)>(45, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Bool, 46 },
        }}),
        // Slot 'onPointCloudVisibilityChanged'
        QtMocHelpers::SlotData<void(bool)>(47, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Bool, 46 },
        }}),
        // Slot 'onLabelVisibilityChanged'
        QtMocHelpers::SlotData<void(int, bool)>(48, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 49 }, { QMetaType::Bool, 46 },
        }}),
        // Slot 'onMaxillaVisibilityChanged'
        QtMocHelpers::SlotData<void(bool)>(50, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Bool, 46 },
        }}),
        // Slot 'onMandibleVisibilityChanged'
        QtMocHelpers::SlotData<void(bool)>(51, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Bool, 46 },
        }}),
        // Slot 'onContactPointsVisibilityChanged'
        QtMocHelpers::SlotData<void(bool)>(52, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Bool, 46 },
        }}),
        // Slot 'updateContactPointsVisualization'
        QtMocHelpers::SlotData<void()>(53, 2, QMC::AccessPrivate, QMetaType::Void),
    };
    QtMocHelpers::UintData qt_properties {
    };
    QtMocHelpers::UintData qt_enums {
    };
    return QtMocHelpers::metaObjectData<MainWindow, qt_meta_tag_ZN10MainWindowE_t>(QMC::MetaObjectFlag{}, qt_stringData,
            qt_methods, qt_properties, qt_enums);
}
Q_CONSTINIT const QMetaObject MainWindow::staticMetaObject = { {
    QMetaObject::SuperData::link<QMainWindow::staticMetaObject>(),
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN10MainWindowE_t>.stringdata,
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN10MainWindowE_t>.data,
    qt_static_metacall,
    nullptr,
    qt_staticMetaObjectRelocatingContent<qt_meta_tag_ZN10MainWindowE_t>.metaTypes,
    nullptr
} };

void MainWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    auto *_t = static_cast<MainWindow *>(_o);
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: _t->openFile(); break;
        case 1: _t->saveFile(); break;
        case 2: _t->toggleWireframe((*reinterpret_cast< std::add_pointer_t<bool>>(_a[1]))); break;
        case 3: _t->resetView(); break;
        case 4: _t->onMeshLoaded((*reinterpret_cast< std::add_pointer_t<size_t>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<size_t>>(_a[2]))); break;
        case 5: _t->runAISegmentation(); break;
        case 6: _t->onSegmentationFinished((*reinterpret_cast< std::add_pointer_t<bool>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<QString>>(_a[2]))); break;
        case 7: _t->onSegmentationProgress((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1]))); break;
        case 8: _t->loadBiteData(); break;
        case 9: _t->roughAlignJaws(); break;
        case 10: _t->runQuickBite(); break;
        case 11: _t->runOptimizeBite(); break;
        case 12: _t->stopOptimization(); break;
        case 13: _t->resetBiteAlignment(); break;
        case 14: _t->onBiteDataLoaded(); break;
        case 15: _t->onOptimizationStep(); break;
        case 16: _t->toggleBeforeAfter((*reinterpret_cast< std::add_pointer_t<bool>>(_a[1]))); break;
        case 17: _t->exportAlignedMeshes(); break;
        case 18: _t->startLandmarkPicking(); break;
        case 19: _t->onLandmarkPicked((*reinterpret_cast< std::add_pointer_t<Landmark>>(_a[1]))); break;
        case 20: _t->onLandmarkPairComplete((*reinterpret_cast< std::add_pointer_t<int>>(_a[1]))); break;
        case 21: _t->applyLandmarkAlignment(); break;
        case 22: _t->cancelLandmarkPicking(); break;
        case 23: _t->onJawMoved(); break;
        case 24: _t->onJawMovedFast(); break;
        case 25: _t->onJawSelectionChanged((*reinterpret_cast< std::add_pointer_t<bool>>(_a[1]))); break;
        case 26: _t->applyManualChanges(); break;
        case 27: _t->onSliderMoved(); break;
        case 28: _t->moveJawByStep((*reinterpret_cast< std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<float>>(_a[2]))); break;
        case 29: _t->onMeshVisibilityChanged((*reinterpret_cast< std::add_pointer_t<bool>>(_a[1]))); break;
        case 30: _t->onPointCloudVisibilityChanged((*reinterpret_cast< std::add_pointer_t<bool>>(_a[1]))); break;
        case 31: _t->onLabelVisibilityChanged((*reinterpret_cast< std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<bool>>(_a[2]))); break;
        case 32: _t->onMaxillaVisibilityChanged((*reinterpret_cast< std::add_pointer_t<bool>>(_a[1]))); break;
        case 33: _t->onMandibleVisibilityChanged((*reinterpret_cast< std::add_pointer_t<bool>>(_a[1]))); break;
        case 34: _t->onContactPointsVisibilityChanged((*reinterpret_cast< std::add_pointer_t<bool>>(_a[1]))); break;
        case 35: _t->updateContactPointsVisualization(); break;
        default: ;
        }
    }
}

const QMetaObject *MainWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *MainWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_staticMetaObjectStaticContent<qt_meta_tag_ZN10MainWindowE_t>.strings))
        return static_cast<void*>(this);
    return QMainWindow::qt_metacast(_clname);
}

int MainWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 36)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 36;
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 36)
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 36;
    }
    return _id;
}
QT_WARNING_POP
