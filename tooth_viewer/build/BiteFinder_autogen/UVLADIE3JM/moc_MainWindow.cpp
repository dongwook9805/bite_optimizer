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
        "runBiteSegmentation",
        "onBiteSegmentationStep",
        "onSegmentationFinished",
        "success",
        "outputPath",
        "onSegmentationProgress",
        "message",
        "loadBiteData",
        "loadExampleData",
        "roughAlignJaws",
        "runQuickBite",
        "runOptimizeBite",
        "stopOptimization",
        "onOptimizationFinished",
        "runCEMOptimization",
        "runESOptimization",
        "runPPOOptimization",
        "savePolicyNetwork",
        "loadPolicyNetwork",
        "resetBiteAlignment",
        "onBiteDataLoaded",
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
        "updateContactPointsVisualization",
        "runOptimizationStep"
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
        // Slot 'runBiteSegmentation'
        QtMocHelpers::SlotData<void()>(12, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onBiteSegmentationStep'
        QtMocHelpers::SlotData<void()>(13, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onSegmentationFinished'
        QtMocHelpers::SlotData<void(bool, const QString &)>(14, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Bool, 15 }, { QMetaType::QString, 16 },
        }}),
        // Slot 'onSegmentationProgress'
        QtMocHelpers::SlotData<void(const QString &)>(17, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::QString, 18 },
        }}),
        // Slot 'loadBiteData'
        QtMocHelpers::SlotData<void()>(19, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'loadExampleData'
        QtMocHelpers::SlotData<void()>(20, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'roughAlignJaws'
        QtMocHelpers::SlotData<void()>(21, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'runQuickBite'
        QtMocHelpers::SlotData<void()>(22, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'runOptimizeBite'
        QtMocHelpers::SlotData<void()>(23, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'stopOptimization'
        QtMocHelpers::SlotData<void()>(24, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onOptimizationFinished'
        QtMocHelpers::SlotData<void()>(25, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'runCEMOptimization'
        QtMocHelpers::SlotData<void()>(26, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'runESOptimization'
        QtMocHelpers::SlotData<void()>(27, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'runPPOOptimization'
        QtMocHelpers::SlotData<void()>(28, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'savePolicyNetwork'
        QtMocHelpers::SlotData<void()>(29, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'loadPolicyNetwork'
        QtMocHelpers::SlotData<void()>(30, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'resetBiteAlignment'
        QtMocHelpers::SlotData<void()>(31, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onBiteDataLoaded'
        QtMocHelpers::SlotData<void()>(32, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'toggleBeforeAfter'
        QtMocHelpers::SlotData<void(bool)>(33, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Bool, 34 },
        }}),
        // Slot 'exportAlignedMeshes'
        QtMocHelpers::SlotData<void()>(35, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'startLandmarkPicking'
        QtMocHelpers::SlotData<void()>(36, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onLandmarkPicked'
        QtMocHelpers::SlotData<void(const Landmark &)>(37, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { 0x80000000 | 38, 39 },
        }}),
        // Slot 'onLandmarkPairComplete'
        QtMocHelpers::SlotData<void(int)>(40, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 41 },
        }}),
        // Slot 'applyLandmarkAlignment'
        QtMocHelpers::SlotData<void()>(42, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'cancelLandmarkPicking'
        QtMocHelpers::SlotData<void()>(43, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onJawMoved'
        QtMocHelpers::SlotData<void()>(44, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onJawMovedFast'
        QtMocHelpers::SlotData<void()>(45, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onJawSelectionChanged'
        QtMocHelpers::SlotData<void(bool)>(46, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Bool, 47 },
        }}),
        // Slot 'applyManualChanges'
        QtMocHelpers::SlotData<void()>(48, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onSliderMoved'
        QtMocHelpers::SlotData<void()>(49, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'moveJawByStep'
        QtMocHelpers::SlotData<void(int, float)>(50, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 51 }, { QMetaType::Float, 52 },
        }}),
        // Slot 'onMeshVisibilityChanged'
        QtMocHelpers::SlotData<void(bool)>(53, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Bool, 54 },
        }}),
        // Slot 'onPointCloudVisibilityChanged'
        QtMocHelpers::SlotData<void(bool)>(55, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Bool, 54 },
        }}),
        // Slot 'onLabelVisibilityChanged'
        QtMocHelpers::SlotData<void(int, bool)>(56, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 57 }, { QMetaType::Bool, 54 },
        }}),
        // Slot 'onMaxillaVisibilityChanged'
        QtMocHelpers::SlotData<void(bool)>(58, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Bool, 54 },
        }}),
        // Slot 'onMandibleVisibilityChanged'
        QtMocHelpers::SlotData<void(bool)>(59, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Bool, 54 },
        }}),
        // Slot 'onContactPointsVisibilityChanged'
        QtMocHelpers::SlotData<void(bool)>(60, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Bool, 54 },
        }}),
        // Slot 'updateContactPointsVisualization'
        QtMocHelpers::SlotData<void()>(61, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'runOptimizationStep'
        QtMocHelpers::SlotData<void()>(62, 2, QMC::AccessPrivate, QMetaType::Void),
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
        case 6: _t->runBiteSegmentation(); break;
        case 7: _t->onBiteSegmentationStep(); break;
        case 8: _t->onSegmentationFinished((*reinterpret_cast< std::add_pointer_t<bool>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<QString>>(_a[2]))); break;
        case 9: _t->onSegmentationProgress((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1]))); break;
        case 10: _t->loadBiteData(); break;
        case 11: _t->loadExampleData(); break;
        case 12: _t->roughAlignJaws(); break;
        case 13: _t->runQuickBite(); break;
        case 14: _t->runOptimizeBite(); break;
        case 15: _t->stopOptimization(); break;
        case 16: _t->onOptimizationFinished(); break;
        case 17: _t->runCEMOptimization(); break;
        case 18: _t->runESOptimization(); break;
        case 19: _t->runPPOOptimization(); break;
        case 20: _t->savePolicyNetwork(); break;
        case 21: _t->loadPolicyNetwork(); break;
        case 22: _t->resetBiteAlignment(); break;
        case 23: _t->onBiteDataLoaded(); break;
        case 24: _t->toggleBeforeAfter((*reinterpret_cast< std::add_pointer_t<bool>>(_a[1]))); break;
        case 25: _t->exportAlignedMeshes(); break;
        case 26: _t->startLandmarkPicking(); break;
        case 27: _t->onLandmarkPicked((*reinterpret_cast< std::add_pointer_t<Landmark>>(_a[1]))); break;
        case 28: _t->onLandmarkPairComplete((*reinterpret_cast< std::add_pointer_t<int>>(_a[1]))); break;
        case 29: _t->applyLandmarkAlignment(); break;
        case 30: _t->cancelLandmarkPicking(); break;
        case 31: _t->onJawMoved(); break;
        case 32: _t->onJawMovedFast(); break;
        case 33: _t->onJawSelectionChanged((*reinterpret_cast< std::add_pointer_t<bool>>(_a[1]))); break;
        case 34: _t->applyManualChanges(); break;
        case 35: _t->onSliderMoved(); break;
        case 36: _t->moveJawByStep((*reinterpret_cast< std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<float>>(_a[2]))); break;
        case 37: _t->onMeshVisibilityChanged((*reinterpret_cast< std::add_pointer_t<bool>>(_a[1]))); break;
        case 38: _t->onPointCloudVisibilityChanged((*reinterpret_cast< std::add_pointer_t<bool>>(_a[1]))); break;
        case 39: _t->onLabelVisibilityChanged((*reinterpret_cast< std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<bool>>(_a[2]))); break;
        case 40: _t->onMaxillaVisibilityChanged((*reinterpret_cast< std::add_pointer_t<bool>>(_a[1]))); break;
        case 41: _t->onMandibleVisibilityChanged((*reinterpret_cast< std::add_pointer_t<bool>>(_a[1]))); break;
        case 42: _t->onContactPointsVisibilityChanged((*reinterpret_cast< std::add_pointer_t<bool>>(_a[1]))); break;
        case 43: _t->updateContactPointsVisualization(); break;
        case 44: _t->runOptimizationStep(); break;
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
        if (_id < 45)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 45;
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 45)
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 45;
    }
    return _id;
}
QT_WARNING_POP
