#include <Python.h>
#include <structmember.h>

#include <numpy/arrayobject.h>
#include "interface/mmal/mmal.h"

typedef struct {
    PyObject_HEAD
    MMAL_COMPONENT_T *camera;
    MMAL_COMPONENT_T *encoder;
    MMAL_ES_FORMAT_T *format;
    MMAL_PORT_T *camera_preview_port;
    MMAL_PORT_T *camera_video_port;
    MMAL_PORT_T *camera_still_port;
    MMAL_POOL_T *pool;

    PyObject *image;
   
} Camera;

static PyMethodDef Camera_methods[] = {
    // {"enable_encoder", (PyCFunction)Camera_enable_encoder, METH_VARARGS,
    //     "Enable/Disable the camera"},
    // {"set_encoder_format", (PyCFunctionWithKeywords)Camera_set_encoder_format, METH_VARARGS | METH_KEYWORDS,
    //     "Set the encoder format"},
    // {"set_camera_format", (PyCFunctionWithKeywords)Camera_set_camera_format, METH_VARARGS | METH_KEYWORDS,
    //     "Set the camera format"},
    // {"set_flash", (PyCFunctionWithKeywords)Camera_set_flash, METH_VARARGS | METH_KEYWORDS,
    //     "Set the camera flash parameters"},
    // {"capture", (PyCFunctionWithKeywords)Camera_capture, METH_VARARGS | METH_KEYWORDS,
    //     "Capture images from camera"},    
    
    {NULL}

};

// static PyMemberDef Camera_members[] = {
//     {"image", T_OBJECT_EX, offsetof(Camera, image), 0, "Numpy image array"},
//     {NULL, NULL, NULL, NULL, NULL}
// };

static PyObject *Camera_get_image(Camera *self, void *closure);
static PyObject *Camera_get_camera_setting(Camera *self, void *closure);
static int Camera_set_camera_setting(Camera *self, PyObject *py_value, void *closure);

static PyGetSetDef Camera_getseters[] = {
    {"image", (getter)Camera_get_image, NULL, "Read-only image buffer", NULL},
    {"saturation", (getter)Camera_get_camera_setting, (setter)Camera_set_camera_setting, "Sensor saturation setting", (void *) MMAL_PARAMETER_SATURATION},
    {"sharpness", (getter)Camera_get_camera_setting, (setter)Camera_set_camera_setting, "Sensor sharpness setting", (void *) MMAL_PARAMETER_SHARPNESS},
    {"contrast", (getter)Camera_get_camera_setting, (setter)Camera_set_camera_setting, "Sensor contrast setting", (void *) MMAL_PARAMETER_CONTRAST},
    {"brightness", (getter)Camera_get_camera_setting, (setter)Camera_set_camera_setting, "Sensor brightness setting", (void *) MMAL_PARAMETER_BRIGHTNESS},
    {"iso", (getter)Camera_get_camera_setting, (setter)Camera_set_camera_setting, "Sensor ISO setting", (void *) MMAL_PARAMETER_ISO},
    {"metering_mode", (getter)Camera_get_camera_setting, (setter)Camera_set_camera_setting, "Sensor metering mode", (void *) MMAL_PARAMETER_EXP_METERING_MODE},
    {"exposure_compensation", (getter)Camera_get_camera_setting, (setter)Camera_set_camera_setting, "Sensor exposure compensation", (void *) MMAL_PARAMETER_CAPTURE_EXPOSURE_COMP},
    {"video_stabilisation", (getter)Camera_get_camera_setting, (setter)Camera_set_camera_setting, "Sensor video stabilisation", (void *) MMAL_PARAMETER_VIDEO_STABILISATION},
    {"exposure_mode", (getter)Camera_get_camera_setting, (setter)Camera_set_camera_setting, "Sensor exposure mode", (void *) MMAL_PARAMETER_EXPOSURE_MODE},
    {"awb_mode", (getter)Camera_get_camera_setting, (setter)Camera_set_camera_setting, "Sensor automatic white-balance mode", (void *) MMAL_PARAMETER_AWB_MODE},
    {"image_fx", (getter)Camera_get_camera_setting, (setter)Camera_set_camera_setting, "Sensor image effects", (void *) MMAL_PARAMETER_IMAGE_EFFECT},
    {NULL}
};



// New, Init, dealloc prototypes
static PyObject * Camera_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
static int Camera_init(Camera *self, PyObject *args, PyObject *kwds);
static void Camera_dealloc (Camera *self);

static PyTypeObject CameraType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "rpicamera.Camera",             /* tp_name */
    sizeof(Camera),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)Camera_dealloc,/* tp_dealloc */
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
    0, //Camera_methods,  /* tp_methods */
    0, //Camera_members,  /* tp_members */
    Camera_getseters,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)Camera_init,      /* tp_init */
    0,                         /* tp_alloc */
    Camera_new,                 /* tp_new */
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
    if (PyType_Ready(&CameraType) < 0)
        return NULL;

    Py_INCREF(&CameraType);
    

    //Import logging module
    PyObject *logging_module = PyImport_ImportModule("logging");

    //Create our own module
    PyObject *m = PyModule_Create(&RpiCameraModule);

    if (m == NULL)
        return NULL;

    //Add Camera Object
    PyModule_AddObject(m, "Camera", (PyObject *)&CameraType);


    //Initialize BCM host board 
    bcm_host_init();

    return m;
}
