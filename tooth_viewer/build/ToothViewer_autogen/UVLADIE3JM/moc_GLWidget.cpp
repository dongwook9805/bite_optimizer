/****************************************************************************
** Meta object code from reading C++ file 'GLWidget.h'
**
** Created by: The Qt Meta Object Compiler version 69 (Qt 6.9.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../src/GLWidget.h"
#include <QtCore/qmetatype.h>

#include <QtCore/qtmochelpers.h>

#include <memory>


#include <QtCore/qxptype_traits.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'GLWidget.h' doesn't include <QObject>."
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
struct qt_meta_tag_ZN8GLWidgetE_t {};
} // unnamed namespace

template <> constexpr inline auto GLWidget::qt_create_metaobjectdata<qt_meta_tag_ZN8GLWidgetE_t>()
{
    namespace QMC = QtMocConstants;
    QtMocHelpers::StringRefStorage qt_stringData {
        "GLWidget",
        "meshLoaded",
        "",
        "size_t",
        "vertices",
        "faces",
        "biteDataLoaded",
        "landmarkPicked",
        "Landmark",
        "landmark",
        "landmarkPairComplete",
        "pairIndex",
        "jawMoved",
        "jawMovedFast",
        "jawSelectionChanged",
        "movingMaxilla"
    };

    QtMocHelpers::UintData qt_methods {
        // Signal 'meshLoaded'
        QtMocHelpers::SignalData<void(size_t, size_t)>(1, 2, QMC::AccessPublic, QMetaType::Void, {{
            { 0x80000000 | 3, 4 }, { 0x80000000 | 3, 5 },
        }}),
        // Signal 'biteDataLoaded'
        QtMocHelpers::SignalData<void()>(6, 2, QMC::AccessPublic, QMetaType::Void),
        // Signal 'landmarkPicked'
        QtMocHelpers::SignalData<void(const Landmark &)>(7, 2, QMC::AccessPublic, QMetaType::Void, {{
            { 0x80000000 | 8, 9 },
        }}),
        // Signal 'landmarkPairComplete'
        QtMocHelpers::SignalData<void(int)>(10, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::Int, 11 },
        }}),
        // Signal 'jawMoved'
        QtMocHelpers::SignalData<void()>(12, 2, QMC::AccessPublic, QMetaType::Void),
        // Signal 'jawMovedFast'
        QtMocHelpers::SignalData<void()>(13, 2, QMC::AccessPublic, QMetaType::Void),
        // Signal 'jawSelectionChanged'
        QtMocHelpers::SignalData<void(bool)>(14, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::Bool, 15 },
        }}),
    };
    QtMocHelpers::UintData qt_properties {
    };
    QtMocHelpers::UintData qt_enums {
    };
    return QtMocHelpers::metaObjectData<GLWidget, qt_meta_tag_ZN8GLWidgetE_t>(QMC::MetaObjectFlag{}, qt_stringData,
            qt_methods, qt_properties, qt_enums);
}
Q_CONSTINIT const QMetaObject GLWidget::staticMetaObject = { {
    QMetaObject::SuperData::link<QOpenGLWidget::staticMetaObject>(),
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN8GLWidgetE_t>.stringdata,
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN8GLWidgetE_t>.data,
    qt_static_metacall,
    nullptr,
    qt_staticMetaObjectRelocatingContent<qt_meta_tag_ZN8GLWidgetE_t>.metaTypes,
    nullptr
} };

void GLWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    auto *_t = static_cast<GLWidget *>(_o);
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: _t->meshLoaded((*reinterpret_cast< std::add_pointer_t<size_t>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<size_t>>(_a[2]))); break;
        case 1: _t->biteDataLoaded(); break;
        case 2: _t->landmarkPicked((*reinterpret_cast< std::add_pointer_t<Landmark>>(_a[1]))); break;
        case 3: _t->landmarkPairComplete((*reinterpret_cast< std::add_pointer_t<int>>(_a[1]))); break;
        case 4: _t->jawMoved(); break;
        case 5: _t->jawMovedFast(); break;
        case 6: _t->jawSelectionChanged((*reinterpret_cast< std::add_pointer_t<bool>>(_a[1]))); break;
        default: ;
        }
    }
    if (_c == QMetaObject::IndexOfMethod) {
        if (QtMocHelpers::indexOfMethod<void (GLWidget::*)(size_t , size_t )>(_a, &GLWidget::meshLoaded, 0))
            return;
        if (QtMocHelpers::indexOfMethod<void (GLWidget::*)()>(_a, &GLWidget::biteDataLoaded, 1))
            return;
        if (QtMocHelpers::indexOfMethod<void (GLWidget::*)(const Landmark & )>(_a, &GLWidget::landmarkPicked, 2))
            return;
        if (QtMocHelpers::indexOfMethod<void (GLWidget::*)(int )>(_a, &GLWidget::landmarkPairComplete, 3))
            return;
        if (QtMocHelpers::indexOfMethod<void (GLWidget::*)()>(_a, &GLWidget::jawMoved, 4))
            return;
        if (QtMocHelpers::indexOfMethod<void (GLWidget::*)()>(_a, &GLWidget::jawMovedFast, 5))
            return;
        if (QtMocHelpers::indexOfMethod<void (GLWidget::*)(bool )>(_a, &GLWidget::jawSelectionChanged, 6))
            return;
    }
}

const QMetaObject *GLWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *GLWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_staticMetaObjectStaticContent<qt_meta_tag_ZN8GLWidgetE_t>.strings))
        return static_cast<void*>(this);
    if (!strcmp(_clname, "QOpenGLFunctions_3_3_Core"))
        return static_cast< QOpenGLFunctions_3_3_Core*>(this);
    return QOpenGLWidget::qt_metacast(_clname);
}

int GLWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QOpenGLWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 7)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 7;
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 7)
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 7;
    }
    return _id;
}

// SIGNAL 0
void GLWidget::meshLoaded(size_t _t1, size_t _t2)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 0, nullptr, _t1, _t2);
}

// SIGNAL 1
void GLWidget::biteDataLoaded()
{
    QMetaObject::activate(this, &staticMetaObject, 1, nullptr);
}

// SIGNAL 2
void GLWidget::landmarkPicked(const Landmark & _t1)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 2, nullptr, _t1);
}

// SIGNAL 3
void GLWidget::landmarkPairComplete(int _t1)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 3, nullptr, _t1);
}

// SIGNAL 4
void GLWidget::jawMoved()
{
    QMetaObject::activate(this, &staticMetaObject, 4, nullptr);
}

// SIGNAL 5
void GLWidget::jawMovedFast()
{
    QMetaObject::activate(this, &staticMetaObject, 5, nullptr);
}

// SIGNAL 6
void GLWidget::jawSelectionChanged(bool _t1)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 6, nullptr, _t1);
}
QT_WARNING_POP
