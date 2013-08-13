/*  Copyright (c) Zachary Berkowitz
    All rights reserved.

    This file is part of the RpiCamera python extension for the
    Raspberry Pi camera module, derived from James Hughes'
    Raspi* family of command-line driven programs which can be found
    at https://github.com/raspberrypi/userland/

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in the
          documentation and/or other materials provided with the distribution.
        * Neither the name of the copyright holder nor the
          names of its contributors may be used to endorse or promote products
          derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
*/

#ifndef RPICAMERA_H_
#define RPICAMERA_H_

#include <Python.h>
#include "structmember.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL RPICAMERA_ARRAY_API
#include <numpy/arrayobject.h>

#include "host_applications/linux/libs/bcm_host/include/bcm_host.h"

#include "RpiCamera_types.h"
#include "RpiCamera_settings.h"
#include "RpiCamera_capture.h"
#include "RpiCamera_logging.h"

PyObject *RPICAMERA_MODULE_LOGGER=NULL;

PyDoc_STRVAR(set_output_format__doc__, 
"set_output_format(channel[, width, height, width_offset, height_offset,\n"
"                           crop_width, crop_height, frame_rate_num,\n"
"                           frame_rate_den, encoding])\n\n"
"Sets the output format for the desired camera port.\n\n\n"
":param channel:  Which port to set the format for.\n"
":type channel:  int.\n\n"
":param width:  Format buffer width.\n"
":type width:  int.\n\n"
":param height:  Format buffer height.\n"
":type height:  int.\n\n"
":param width_offset:  Format sensor horizontal offset.\n"
":type width_offset:  int.\n\n"
":param height_offset:  Format sensor vertical offset.\n"
":type height_offset:  int.\n\n"
":param crop_width:  Format width.\n"
":type crop_width:  int.\n\n"
":param crop_height:  Format height.\n"
":type crop_height:  int.\n\n"
":param frame_rate_num:  Numerator of the output frame rate.\n"
":type frame_rate_num:  int.\n\n"
":param frame_rate_den:  Denominator of the output frame rate.\n"
":type frame_rate_den:  int.\n\n"
":param encoding:  Encoding.\n"
":type encoding:  int.\n\n\n"
"Camera channels:\n"
"   Use following constants to set the output channel:\n"
"   RpiCamera.CAMERA_STILL      Use the still capture output.\n"
"   RpiCamera.CAMERA_PREVIEW    Use the preview output.\n\n"
"Image geometry:\n"
"   Total size (in pixels) of the image read is width*height.  However,\n"
"   the actual image data (valid pixels) will be crop_width*crop_height,\n"
"   with top-right corner offset of (width_offset, height_offset)\n\n"
"Framerates:\n"
"   Framerates are encoded as rational numbers, with the numerator and denominator\n"
"   listed speparately.  In most cases we are interested in integer framerates, for\n"
"   example, 20 FPS would have a numerator of 20 and a denominator of 1\n\n"
"Encoding:\n"
"   Two encoding options for stills are available:  YUV_420I and BGR24.\n"
"   Set the encoding type using the constants:  RpiCamera.YUV_420I and RpiCamera.BGR24\n\n"
"   For an image size of HxW, a YUV_420I image will have a total of\n"
"   1.5*H*W bytes of data, the BGR24 image will have 3*H*W.  For details\n"
"   on each format, see:\n"
"   http://en.wikipedia.org/wiki/YUV#Y.27UV420p_.28and_Y.27V12_or_YV12.29_to_RGB888_conversion\n"
"   http://msdn.microsoft.com/en-us/library/system.windows.media.pixelformats.bgr24.aspx.");
static PyObject *RpiCamera_set_output_format(RpiCamera *self, PyObject *args, PyObject *kwds);

PyDoc_STRVAR(get_output_format__doc__, 
"get_output_format(channel)\n\n"
"Gets the output format for the desired camera port.\n\n\n"
":param channel:  Which port to set the format for.\n"
":type channel:  int.\n\n"
":returns format:  Dictionary with the format specification (see RpiCamera.Camera.set_output_format).\n");
static PyObject *RpiCamera_get_output_format(RpiCamera *self, PyObject *args, PyObject *kwds);


PyDoc_STRVAR(switch_output__doc__,
"switch_output(channel)\n\n"
"Switches the output channel used for capturing still images with\n"
"RpiCamera.Camera.capture_still_image.\n\n"
"Camera channels:\n"
"   Use following constants to set the output channel:\n"
"   RpiCamera.CAMERA_STILL      Use the still capture output.\n"
"   RpiCamera.CAMERA_PREVIEW    Use the preview output.\n\n");
PyObject *RpiCamera_switch_output(RpiCamera *self, PyObject *args);

static PyMethodDef RpiCamera_methods[] = {
    {"set_output_format", (PyCFunction)RpiCamera_set_output_format, METH_VARARGS | METH_KEYWORDS, set_output_format__doc__},
    {"get_output_format", (PyCFunction)RpiCamera_get_output_format, METH_VARARGS | METH_KEYWORDS, get_output_format__doc__},
    {"integrate_preview_frames", (PyCFunction)RpiCamera_integrated_preview, METH_VARARGS | METH_KEYWORDS, integrate_preview_frames__doc__},
    {"capture_still_image", (PyCFunction)RpiCamera_capture_still, METH_NOARGS, capture_still_image__doc__},
    {"switch_output_port", (PyCFunction)RpiCamera_switch_output, METH_VARARGS, switch_output__doc__},
    {NULL}

};

static PyMemberDef RpiCamera_members[] = {
    {"debug", T_BOOL, offsetof(RpiCamera, debug_flag), 0, "Enable debug message logging"},
    {NULL}
};

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
    {"flash_mode", (getter)RpiCamera_get_camera_setting, (setter)RpiCamera_set_camera_setting, "Camera flash mode", (void *) MMAL_PARAMETER_FLASH},
    {"flash_type", (getter)RpiCamera_get_camera_setting, (setter)RpiCamera_set_camera_setting, "Camera flash type", (void *) MMAL_PARAMETER_FLASH_SELECT},
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
    RpiCamera_members,  /* tp_members */
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

// void Free_RpiCameraModule(void){
//     if (logging_semaphore)
//         vcos_semaphore_delete(logging_semaphore);
// }

static PyModuleDef RpiCameraModule = {
    PyModuleDef_HEAD_INIT,
    "rpicamera",
    "Python module for use witht he RPi Camera",
    -1,
    NULL, NULL, NULL, NULL, NULL //Free_RpiCameraModule
};

PyMODINIT_FUNC
PyInit_RpiCamera(void){

    // vcos_semaphore_create(logging_semaphore, "log_semaphore", 1);

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

    //Add some constants
    PyModule_AddObject(m, "CAMERA_STILL", PyLong_FromLong(MMAL_CAMERA_CAPTURE_PORT));
    PyModule_AddObject(m, "CAMERA_VIDEO", PyLong_FromLong(MMAL_CAMERA_VIDEO_PORT));
    PyModule_AddObject(m, "CAMERA_PREVIEW", PyLong_FromLong(MMAL_CAMERA_PREVIEW_PORT));
    PyModule_AddObject(m, "YUV_420I", PyLong_FromLong(MMAL_ENCODING_I420));
    PyModule_AddObject(m, "BGR24", PyLong_FromLong(MMAL_ENCODING_BGR24));

    //Initialize BCM host board 
    bcm_host_init();

    return m;
}
#endif