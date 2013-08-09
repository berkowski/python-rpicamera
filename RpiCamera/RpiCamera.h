#ifndef RPICAMERA_H_
#define RPICAMERA_H_

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL RPICAMERA_ARRAY_API
#include <numpy/arrayobject.h>

#include "RpiCamera_types.h"
#include "RpiCamera_settings.h"
#include "RpiCamera_capture.h"
#include "RpiCamera_logging.h"
PyObject *RPICAMERA_MODULE_LOGGER=NULL;


static PyMethodDef RpiCamera_methods[] = {
    {"capture_still_frames", (PyCFunctionWithKeywords)RpiCamera_capture_stills, METH_VARARGS | METH_KEYWORDS,
        "Capture still frame(s) using the currently loaded format."},
    {NULL, NULL, NULL, NULL}

};

// static PyMemberDef Camera_members[] = {
//     {"image", T_OBJECT_EX, offsetof(Camera, image), 0, "Numpy image array"},
//     {NULL, NULL, NULL, NULL, NULL}
// };

static PyObject *RpiCamera_get_image(RpiCamera *self, void *closure);
static PyObject *RpiCamera_get_camera_setting(RpiCamera *self, void *closure);
static int RpiCamera_set_camera_setting(RpiCamera *self, PyObject *py_value, void *closure);

static PyGetSetDef RpiCamera_getseters[] = {
    {"image", (getter)RpiCamera_get_image, NULL, "Read-only image buffer", NULL},
    {"saturation", (getter)RpiCamera_get_camera_setting, (setter)RpiCamera_set_camera_setting, "Sensor saturation setting", (void *) MMAL_PARAMETER_SATURATION},
    {"sharpness", (getter)RpiCamera_get_camera_setting, (setter)RpiCamera_set_camera_setting, "Sensor sharpness setting", (void *) MMAL_PARAMETER_SHARPNESS},
    {"contrast", (getter)RpiCamera_get_camera_setting, (setter)RpiCamera_set_camera_setting, "Sensor contrast setting", (void *) MMAL_PARAMETER_CONTRAST},
    {"brightness", (getter)RpiCamera_get_camera_setting, (setter)RpiCamera_set_camera_setting, "Sensor brightness setting", (void *) MMAL_PARAMETER_BRIGHTNESS},
    {"iso", (getter)RpiCamera_get_camera_setting, (setter)RpiCamera_set_camera_setting, "Sensor ISO setting", (void *) MMAL_PARAMETER_ISO},
    {"metering_mode", (getter)RpiCamera_get_camera_setting, (setter)RpiCamera_set_camera_setting, "Sensor metering mode", (void *) MMAL_PARAMETER_EXP_METERING_MODE},
    {"exposure_compensation", (getter)RpiCamera_get_camera_setting, (setter)RpiCamera_set_camera_setting, "Sensor exposure compensation", (void *) MMAL_PARAMETER_CAPTURE_EXPOSURE_COMP},
    {"video_stabilisation", (getter)RpiCamera_get_camera_setting, (setter)RpiCamera_set_camera_setting, "Sensor video stabilisation", (void *) MMAL_PARAMETER_VIDEO_STABILISATION},
    {"exposure_mode", (getter)RpiCamera_get_camera_setting, (setter)RpiCamera_set_camera_setting, "Sensor exposure mode", (void *) MMAL_PARAMETER_EXPOSURE_MODE},
    {"awb_mode", (getter)RpiCamera_get_camera_setting, (setter)RpiCamera_set_camera_setting, "Sensor automatic white-balance mode", (void *) MMAL_PARAMETER_AWB_MODE},
    {"image_fx", (getter)RpiCamera_get_camera_setting, (setter)RpiCamera_set_camera_setting, "Sensor image effects", (void *) MMAL_PARAMETER_IMAGE_EFFECT},
    {NULL}
};



// New, Init, dealloc prototypes
static PyObject * RpiCamera_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
static int RpiCamera_init(RpiCamera *self, PyObject *args, PyObject *kwds);
static void RpiCamera_dealloc (RpiCamera *self);

static PyTypeObject RpiCameraType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "rpicamera.Camera",             /* tp_name */
    sizeof(RpiCamera),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)RpiCamera_dealloc,/* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT |
        Py_TPFLAGS_BASETYPE,   /* tp_flags */
    "Camera object for the Raspberry Pi camera module", /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    RpiCamera_methods,  /* tp_methods */
    0, //Camera_members,  /* tp_members */
    RpiCamera_getseters,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)RpiCamera_init,      /* tp_init */
    0,                         /* tp_alloc */
    RpiCamera_new,                 /* tp_new */
};

static PyModuleDef RpiCameraModule = {
    PyModuleDef_HEAD_INIT,
    "rpicamera",
    "Python module for use witht he RPi Camera",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit_RpiCamera(void){

    import_array();
    if (PyType_Ready(&RpiCameraType) < 0)
        return NULL;

    Py_INCREF(&RpiCameraType);
    

    //Import logging module
    PyObject *logging_module = PyImport_ImportModule("logging");
    RPICAMERA_MODULE_LOGGER = PyObject_CallMethod(logging_module, "getLogger", "s", "RpiCamera");
    Py_INCREF(RPICAMERA_MODULE_LOGGER);

    //Create our own module
    PyObject *m = PyModule_Create(&RpiCameraModule);

    if (m == NULL)
        return NULL;

    //Add Camera Object
    PyModule_AddObject(m, "Camera", (PyObject *)&RpiCameraType);
    PyModule_AddObject(m, "logger", RPICAMERA_MODULE_LOGGER);

    //Initialize BCM host board 
    bcm_host_init();

    return m;
}
#endif