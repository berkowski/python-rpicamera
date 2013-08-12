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
#include "interface/mmal/util/mmal_default_components.h"
#include "interface/mmal/util/mmal_util_params.h"
#include "interface/mmal/util/mmal_util.h"
#include "interface/mmal/mmal_pool.h"
#include "interface/vcos/vcos_types.h"
#include "interface/vcos/vcos_semaphore.h"

#include "RpiCamera.h"
#include "RpiCamera_capture.h"

// Stills format information
#define DEFAULT_STILLS_FRAME_RATE_NUM 3
#define DEFAULT_STILLS_FRAME_RATE_DEN 1
#define DEFAULT_STILLS_MAX_WIDTH 2624
#define DEFAULT_STILLS_MAX_HEIGHT 1956
#define DEFAULT_STILLS_CROP_X_OFFSET 16
#define DEFAULT_STILLS_CROP_WIDTH 2592
#define DEFAULT_STILLS_CROP_Y_OFFSET 6
#define DEFAULT_STILLS_CROP_HEIGHT 1944

#define DEFAULT_PREVIEW_FRAME_RATE_NUM 30
#define DEFAULT_PREVIEW_FRAME_RATE_DEN  1
#define DEFAULT_PREVIEW_WIDTH 1024
#define DEFAULT_PREVIEW_HEIGHT 768

#define DEFAULT_VIDEO_OUTPUT_BUFFERS_NUM 3

/*  Private helper functions */


void __camera_control_callback(MMAL_PORT_T *port, MMAL_BUFFER_HEADER_T *buffer){
    // fprintf(stderr, "Camera control callback  cmd=0x%08x", buffer->cmd);

    if (buffer->cmd == MMAL_EVENT_PARAMETER_CHANGED){
    }
    else{
        rpicamera_log_error("Received unexpected camera control callback event, 0x%08x", buffer->cmd);
    }

    mmal_buffer_header_release(buffer);
}

MMAL_STATUS_T __set_default_camera_parameters(MMAL_COMPONENT_T *camera){
    MMAL_STATUS_T status = MMAL_SUCCESS;
    const MMAL_PARAMETER_COLOURFX_T color_fx = {{MMAL_PARAMETER_COLOUR_EFFECT, sizeof(color_fx)},
                                                 MMAL_FALSE, 0, 0};

    status |= set_camera_saturation(camera, 0);
    status |= set_camera_sharpness(camera, 0);
    status |= set_camera_contrast(camera, 0);
    status |= set_camera_brightness(camera, 50);
    status |= set_camera_iso(camera, 400);
    status |= set_camera_video_stabilisation(camera, MMAL_FALSE);
    status |= set_camera_exposure_compensation(camera, 0);
    status |= set_camera_exposure_mode(camera, MMAL_PARAM_EXPOSUREMODE_AUTO);
    status |= set_camera_metering_mode(camera, MMAL_PARAM_EXPOSUREMETERINGMODE_MATRIX);
    status |= set_camera_awb_mode(camera, MMAL_PARAM_AWBMODE_AUTO);
    status |= set_camera_image_fx(camera, MMAL_PARAM_IMAGEFX_NONE);
    status |= set_camera_colour_fx(camera, &color_fx);
    status |= set_camera_rotation(camera, 0);
    status |= set_camera_flips(camera, MMAL_PARAM_MIRROR_NONE);

    return status;
    // status |= raspicamcontrol_set_thumbnail_parameters(camera, &params->thumbnailConfig);  TODO Not working for some reason
}

MMAL_STATUS_T __setup_camera(RpiCamera *RpiCamera){

    MMAL_STATUS_T status = MMAL_SUCCESS;
    MMAL_COMPONENT_T *camera = NULL;
    MMAL_ES_FORMAT_T *format;
    MMAL_PORT_T *preview_port, *video_port, *still_port;

    if (!RpiCamera)
        // Camera object needs to be initialized first
        goto error;

    camera = RpiCamera->camera;

    status = mmal_component_create(MMAL_COMPONENT_DEFAULT_CAMERA, &camera);

    if (status != MMAL_SUCCESS)
    {
      rpicamera_log_error("Failed to create camera component", NULL);
      goto error;
    }

    if (!camera->output_num)
    {
      rpicamera_log_error("Camera doesn't have output ports", NULL);
      goto error;
    }

    // Enable the camera, and tell it its control callback function
    status = mmal_port_enable(camera->control, __camera_control_callback);

    if (status)
    {
      rpicamera_log_error("Unable to enable control port : error %d", status);
      goto error;
    }

    //  set up the camera configuration
    {
      MMAL_PARAMETER_CAMERA_CONFIG_T cam_config =
        {
            {MMAL_PARAMETER_CAMERA_CONFIG, sizeof(cam_config)},
            .max_stills_w = DEFAULT_STILLS_MAX_WIDTH,
            .max_stills_h = DEFAULT_STILLS_MAX_HEIGHT,
            .stills_yuv422 = 0,
            .one_shot_stills = 1,
            .max_preview_video_w = DEFAULT_PREVIEW_WIDTH,
            .max_preview_video_h = DEFAULT_PREVIEW_HEIGHT,
            .num_preview_video_frames = 3,
            .stills_capture_circular_buffer_height = 0,
            .fast_preview_resume = 0,
            .use_stc_timestamp = MMAL_PARAM_TIMESTAMP_MODE_RESET_STC
            };
        status = mmal_port_parameter_set(camera->control, &cam_config.hdr);
    }

    if (status){
        rpicamera_log_error("Unable to set camera config", NULL);
        goto error;
    }

    status = __set_default_camera_parameters(camera);

    if (status){
        rpicamera_log_error("Unable to set default camera parameters", NULL);
        goto error;
    }

    preview_port = camera->output[MMAL_CAMERA_PREVIEW_PORT];
    format = preview_port->format;

    format->encoding = MMAL_ENCODING_I420;
    format->encoding_variant = MMAL_ENCODING_I420;

    format->es->video.width = DEFAULT_PREVIEW_WIDTH;
    format->es->video.height = DEFAULT_PREVIEW_HEIGHT;
    format->es->video.crop.x = 0;
    format->es->video.crop.y = 0;
    format->es->video.crop.width = DEFAULT_PREVIEW_WIDTH;
    format->es->video.crop.height = DEFAULT_PREVIEW_HEIGHT;
    format->es->video.frame_rate.num = DEFAULT_PREVIEW_FRAME_RATE_NUM;
    format->es->video.frame_rate.den = DEFAULT_PREVIEW_FRAME_RATE_DEN;

    status = mmal_port_format_commit(preview_port);

    if (status)
    {
      rpicamera_log_error("camera viewfinder format couldn't be set", NULL);
      goto error;
    }

    video_port = camera->output[MMAL_CAMERA_VIDEO_PORT];
    mmal_format_full_copy(video_port->format, format);

    status = mmal_port_format_commit(video_port);

    if (status){
      rpicamera_log_error("camera video format couldn't be set", NULL);
      goto error;
    }

    // Ensure there are enough buffers to avoid dropping frames
    if (video_port->buffer_num < DEFAULT_VIDEO_OUTPUT_BUFFERS_NUM)
      video_port->buffer_num = DEFAULT_VIDEO_OUTPUT_BUFFERS_NUM;

    still_port = camera->output[MMAL_CAMERA_CAPTURE_PORT];
    format = still_port->format;

    format->encoding = MMAL_ENCODING_I420;
    format->encoding_variant = MMAL_ENCODING_I420;
    
    format->es->video.width = DEFAULT_STILLS_MAX_WIDTH;
    format->es->video.height = DEFAULT_STILLS_MAX_HEIGHT;
    format->es->video.crop.x = DEFAULT_STILLS_CROP_X_OFFSET;
    format->es->video.crop.y = DEFAULT_STILLS_CROP_Y_OFFSET;
    format->es->video.crop.width = DEFAULT_STILLS_CROP_WIDTH;
    format->es->video.crop.height = DEFAULT_STILLS_CROP_HEIGHT;
    format->es->video.frame_rate.num = DEFAULT_STILLS_FRAME_RATE_NUM;
    format->es->video.frame_rate.den = DEFAULT_STILLS_FRAME_RATE_DEN;
    
    if (still_port->buffer_size < still_port->buffer_size_min)
      still_port->buffer_size = still_port->buffer_size_min;

    still_port->buffer_num = still_port->buffer_num_recommended;

    status = mmal_port_format_commit(still_port);

    if (status)
    {
      rpicamera_log_error("camera still format couldn't be set", NULL);
      goto error;
    }

    /* Enable component */
    status = mmal_component_enable(camera);

    if (status)
    {
      rpicamera_log_error("camera component couldn't be enabled", NULL);
      goto error;
    }
    
    RpiCamera->camera = camera;

    return status;

error:

    if (camera)
        mmal_component_destroy(camera);

    return status;

}

// New, Init, dealloc prototypes
static PyObject *
RpiCamera_new(PyTypeObject *type, PyObject *args, PyObject *kwds){

    RpiCamera *self;
    int img_dim = 0;

    self = (RpiCamera *)type->tp_alloc(type, 0);
    if (self == NULL)
        return (PyObject *)self;
    
    self->image = PyArray_FromDims(1, &img_dim, NPY_UINT8);
    Py_INCREF(self->image);
  
    return (PyObject *)self;

}



static int 
RpiCamera_init(RpiCamera *self, PyObject *args, PyObject *kwds){

    MMAL_STATUS_T status;
    MMAL_PORT_T *port;
    // MMAL_ES_FORMAT_T *format;
    // VCOS_STATUS_T vcos_status = VCOS_SUCCESS;

    status = __setup_camera(self);

    if (status != MMAL_SUCCESS)
        return (int)status;

    self->output_port = MMAL_CAMERA_PREVIEW_PORT;
    port = self->camera->output[MMAL_CAMERA_PREVIEW_PORT];

    self->pool = mmal_port_pool_create(port, port->buffer_num, port->buffer_size);
    self->complete_semaphore = (VCOS_SEMAPHORE_T *) malloc(sizeof(VCOS_SEMAPHORE_T));

    if(self->complete_semaphore == NULL){
        rpicamera_log_fatal("Unable to alloc memory for completion semaphore",  NULL);
        return -1;
    }

    if (vcos_semaphore_create(self->complete_semaphore, "RpiCamera-sem", 0) != VCOS_SUCCESS){
        rpicamera_log_fatal("Unable to create completion semaphore.", NULL);
        return -1;
    }

    return 0;
}

static void 
RpiCamera_dealloc (RpiCamera *self){

    Py_XDECREF(self->image);

    if (self->camera)
        mmal_component_destroy(self->camera);

    if (self->pool)
        mmal_port_pool_destroy(self->camera->output[self->output_port], self->pool);

    if (self->complete_semaphore){
        vcos_semaphore_delete(self->complete_semaphore);
        free(self->complete_semaphore);
    }

    Py_TYPE(self)->tp_free((PyObject *)self);
}

/*  Getters & Setters

*/
static PyObject *
RpiCamera_get_image(RpiCamera *self, void *closure){
    Py_INCREF(self->image);
    return self->image;
}

static PyObject *
RpiCamera_set_output_format(RpiCamera *self, PyObject *args, PyObject *kwds){

    static char *kwlist[] = {"channel", "width", "height", "width_offset", "height_offset",
                             "crop_width", "crop_height", "frame_rate_num", "frame_rate_den", 
                             "encoding", NULL};

    uint8_t channel = MMAL_CAMERA_CAPTURE_PORT;
    uint32_t width = DEFAULT_STILLS_MAX_WIDTH;
    uint32_t height = DEFAULT_STILLS_MAX_HEIGHT;
    uint32_t width_offset = 0;
    uint32_t height_offset = 0;
    uint32_t crop_width = 0;
    uint32_t crop_height = 0;
    uint32_t frame_rate_num = 15;
    uint32_t frame_rate_den = 1;
    uint32_t encoding = MMAL_ENCODING_I420;


    MMAL_ES_FORMAT_T *format;
    MMAL_PORT_T *port;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|BIIIIIIIII", kwlist, &channel, &width, 
                                        &height, &width_offset, &height_offset, &crop_width, 
                                        &crop_height, &frame_rate_num, &frame_rate_den, &encoding))
        return NULL;


    if (!crop_width)
        crop_width = width - width_offset;

    if (!crop_height)
        crop_height = height - height_offset;


    if ((channel < 0) || (channel > 2)){
        PyErr_Format(PyExc_ValueError, "Invalid output channel, choose 0, 1, or 2");
        return NULL;
    }

    if (width > DEFAULT_STILLS_MAX_WIDTH){
        PyErr_Format(PyExc_ValueError, "Width exceeded maximum allowed (%d > %d)", width, DEFAULT_STILLS_MAX_WIDTH);
        return NULL;
    }

    if (height > DEFAULT_STILLS_MAX_HEIGHT){
        PyErr_Format(PyExc_ValueError, "Height exceeded maximum allowed (%d > %d)", height, DEFAULT_STILLS_MAX_HEIGHT);
        return NULL;
    }

    if (crop_width + width_offset > width){
        PyErr_Format(PyExc_ValueError, "Crop width + offset exceeded maximum allowed (%d > %d)", crop_width + width_offset, width);
        return NULL;
    }

    if (crop_height + height_offset > DEFAULT_STILLS_MAX_HEIGHT){
        PyErr_Format(PyExc_ValueError, "Crop height + offset exceeded maximum allowed (%d > %d)", crop_height + height_offset, height);
        return NULL;
    }


    if (crop_width + width_offset > width){
        PyErr_Format(PyExc_ValueError, "Crop Height exceeded maximum allowed (%d > %d)", crop_width + width_offset, width);
        return NULL;        
    }

    switch (encoding){
        case MMAL_ENCODING_I420:
        case MMAL_ENCODING_BGR24:
            break;

        default:
            PyErr_Format(PyExc_ValueError, "Unsupported encoding id:  %d", encoding);
            return NULL;
    }

    port = self->camera->output[channel];
    
    if(port->is_enabled){
        rpicamera_log_debug("Disabling output port prior to setting format", NULL);
        mmal_port_disable(port);
    }

    format = port->format;

    format->encoding = encoding;
    format->encoding_variant = encoding;

    format->es->video.width = (uint32_t)width;
    format->es->video.height = (uint32_t)height;
    format->es->video.crop.x = (int32_t)width_offset;
    format->es->video.crop.y = (int32_t)height_offset;
    format->es->video.crop.width = (int32_t)crop_width;
    format->es->video.crop.height = (int32_t)crop_height;
    format->es->video.frame_rate.num = (int32_t)frame_rate_num;
    format->es->video.frame_rate.den = (int32_t)frame_rate_den;

    if (port->buffer_size < port->buffer_size_min)
      port->buffer_size = port->buffer_size_min;

    port->buffer_num = port->buffer_num_recommended;

    rpicamera_log_debug("Creating new buffer pool...", NULL);
    rpicamera_log_debug("     Destroying old pool...", NULL);
    mmal_port_pool_destroy(port, self->pool);
    self->pool = (MMAL_POOL_T *)NULL;
    
    rpicamera_log_debug("     Creating new pool...", NULL);
    self->pool = mmal_port_pool_create(port, port->buffer_num, port->buffer_size);
    
    free(self->complete_semaphore);
    self->complete_semaphore = (VCOS_SEMAPHORE_T *) malloc(sizeof(VCOS_SEMAPHORE_T));

    if(!self->pool){
        PyErr_SetString(PyExc_RuntimeError, "Unable to create new camera pool.");
        return NULL;        
    }

    if(mmal_port_format_commit(port) != MMAL_SUCCESS){
        PyErr_SetString(PyExc_RuntimeError, "Unable to commit format to camera output port");
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject *
RpiCamera_get_output_format(RpiCamera *self, PyObject *args, PyObject *kwds){

    static char *kwlist[] = {"channel", NULL};
    uint8_t channel = MMAL_CAMERA_CAPTURE_PORT;
    int8_t status = 0;
    MMAL_ES_FORMAT_T *format;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|B", kwlist, &channel))
        return NULL;

    if ((channel < 0) || (channel > 2)){
        PyErr_Format(PyExc_ValueError, "Invalid output channel, choose 0, 1, or 2");
        return NULL;
    }

    PyObject *d = PyDict_New();

    if (!d){
        PyErr_Format(PyExc_RuntimeError, "Unable to create format dictionary object");
        return NULL;
    }
    
    format = self->camera->output[channel]->format;

    status |= PyDict_SetItemString(d, "width", PyLong_FromUnsignedLong((unsigned long)format->es->video.width));
    status |= PyDict_SetItemString(d, "height", PyLong_FromUnsignedLong((unsigned long)format->es->video.height));
    status |= PyDict_SetItemString(d, "width_offset", PyLong_FromUnsignedLong((unsigned long)format->es->video.crop.x));
    status |= PyDict_SetItemString(d, "height_offset", PyLong_FromUnsignedLong((unsigned long)format->es->video.crop.y));
    status |= PyDict_SetItemString(d, "crop_width", PyLong_FromUnsignedLong((unsigned long)format->es->video.crop.width));
    status |= PyDict_SetItemString(d, "crop_height", PyLong_FromUnsignedLong((unsigned long)format->es->video.crop.height));
    status |= PyDict_SetItemString(d, "frame_rate_num", PyLong_FromUnsignedLong((unsigned long)format->es->video.frame_rate.num));
    status |= PyDict_SetItemString(d, "frame_rate_den", PyLong_FromUnsignedLong((unsigned long)format->es->video.frame_rate.den));
    status |= PyDict_SetItemString(d, "encoding", PyLong_FromUnsignedLong((unsigned long)format->encoding));

    if(status){
        PyErr_Format(PyExc_RuntimeError, "Unable to populate format dictionary");
        return NULL;        
    }
    Py_INCREF(d);
    return d;
}

static PyObject *
RpiCamera_get_camera_setting(RpiCamera *self, void *closure){

    int32_t param = (int32_t)closure;
    int32_t value;


    PyObject *result;
    
    // PyObject *dict, *tuple;
    // MMAL_PARAMETER_CAMERA_INFO_T cam_info;
    // uint8_t k;

    MMAL_STATUS_T status = MMAL_SUCCESS;

    switch(param){
        case MMAL_PARAMETER_SATURATION:
            status = get_camera_saturation(self->camera, &value);
            result = PyLong_FromLong((long)value);
            break;

        case MMAL_PARAMETER_SHARPNESS:
            status = get_camera_sharpness(self->camera, &value);
            result = PyLong_FromLong((long)value);
            break;

        case MMAL_PARAMETER_CONTRAST:
            status = get_camera_contrast(self->camera, &value);
            result = PyLong_FromLong((long)value);
            break;

        case MMAL_PARAMETER_BRIGHTNESS:
            status = get_camera_brightness(self->camera, &value);
            result = PyLong_FromLong((long)value);
            break;

        case MMAL_PARAMETER_ISO:
            status = get_camera_iso(self->camera, (uint32_t *)&value);
            result = PyLong_FromLong((long)value);
            break;

        case MMAL_PARAMETER_EXP_METERING_MODE:
            status = get_camera_metering_mode(self->camera, (MMAL_PARAM_EXPOSUREMETERINGMODE_T *)&value);
            result = PyLong_FromLong((long)value);
            break;

        case MMAL_PARAMETER_CAPTURE_EXPOSURE_COMP:
            status = get_camera_exposure_compensation(self->camera, &value);
            result = PyLong_FromLong((long)value);
            break;

        case MMAL_PARAMETER_VIDEO_STABILISATION:
            status = get_camera_video_stabilisation(self->camera, (MMAL_BOOL_T *)&value);
            result = PyBool_FromLong((long)value);
            break;

        case MMAL_PARAMETER_EXPOSURE_MODE:
            status = get_camera_exposure_mode(self->camera, (MMAL_PARAM_EXPOSUREMODE_T *)&value);
            result = PyLong_FromLong((long)value);
            break;

        case MMAL_PARAMETER_AWB_MODE:
            status = get_camera_awb_mode(self->camera, (MMAL_PARAM_AWBMODE_T *)&value);
            result = PyLong_FromLong((long)value);
            break;

        case MMAL_PARAMETER_IMAGE_EFFECT:
            status = get_camera_image_fx(self->camera, (MMAL_PARAM_IMAGEFX_T *)&value);
            result = PyLong_FromLong((long)value);
            break;

        case MMAL_PARAMETER_FLASH:
            status = get_camera_flash_mode(self->camera, (MMAL_PARAM_FLASH_T *)&value);
            result = PyLong_FromLong((long)value);
            break;

        case MMAL_PARAMETER_FLASH_SELECT:
            status = get_camera_flash_type(self->camera, (MMAL_PARAMETER_CAMERA_INFO_FLASH_TYPE_T *)&value);
            result = PyLong_FromLong((long)value);
            break;

        // case MMAL_PARAMETER_CAMERA_INFO:
        //     status = get_camera_info(self->camera, &cam_info);
        //     if (status != MMAL_SUCCESS){
        //         PyErr_SetString(PyExc_RuntimeError, "Unable to get camera info");
        //         break;
        //     }

            
        //     tuple = PyTuple_New((Py_ssize_t) cam_info.num_cameras);
        //     result = PyDict_New();

        //     for (k=0; k<cam_info.num_cameras; k++){
        //         dict = PyDict_New();

        //         PyDict_SetItemString(dict, "port_id", PyLong_FromUnsignedLong((unsigned long)cam_info.cameras[k].port_id));
        //         PyDict_SetItemString(dict, "max_height", PyLong_FromUnsignedLong((unsigned long)cam_info.cameras[k].max_height));
        //         PyDict_SetItemString(dict, "max_width", PyLong_FromUnsignedLong((unsigned long)cam_info.cameras[k].max_width));
        //         PyDict_SetItemString(dict, "lens_present", PyBool_FromLong((long)cam_info.cameras[k].lens_present));

        //         PyTuple_SetItem(tuple, (Py_ssize_t)k, dict);
        //         Py_INCREF(dict);
        //     }

        //     PyDict_SetItemString(result, "cameras", tuple);
        //     tuple = PyTuple_New((Py_ssize_t) cam_info.num_flashes);
            
        //     for (k=0; k<cam_info.num_flashes; k++){
        //         dict = PyLong_FromUnsignedLong((unsigned long) cam_info.flashes[k].flash_type);
        //         PyTuple_SetItem(tuple, (Py_ssize_t)k, dict);
        //     }            

        //     PyDict_SetItemString(result, "flashes", tuple);
        //     break;

        default:
            PyErr_Format(PyExc_RuntimeError, "Unknown parameter id:  %d", param);
            return NULL;
    }

    if (status != MMAL_SUCCESS){
        PyErr_SetString(PyExc_AttributeError, "Unable to read attribute from camera");
        return NULL;
    }

    Py_INCREF(result);
    return result;
}

static int
RpiCamera_set_camera_setting(RpiCamera *self, PyObject *py_value, void *closure){

    int32_t param = (int32_t)closure;
    int32_t value;

    MMAL_STATUS_T status = MMAL_SUCCESS;

    if (!PyLong_Check(py_value)){
        PyErr_SetString(PyExc_ValueError, "Expected an integer.");
        return -1;
    }

    value = (int32_t) PyLong_AsLong(py_value);

    switch(param){
        case MMAL_PARAMETER_SATURATION:
            status = set_camera_saturation(self->camera, value);
            break;

        case MMAL_PARAMETER_SHARPNESS:
            status = set_camera_sharpness(self->camera, value);
            break;

        case MMAL_PARAMETER_CONTRAST:
            status = set_camera_contrast(self->camera, value);
            break;

        case MMAL_PARAMETER_BRIGHTNESS:
            status = set_camera_brightness(self->camera, value);
            break;

        case MMAL_PARAMETER_ISO:
            status = set_camera_iso(self->camera, (uint32_t)value);
            break;

        case MMAL_PARAMETER_EXP_METERING_MODE:
            status = set_camera_metering_mode(self->camera, (MMAL_PARAM_EXPOSUREMETERINGMODE_T)value);
            break;

        case MMAL_PARAMETER_CAPTURE_EXPOSURE_COMP:
            status = set_camera_exposure_compensation(self->camera, value);
            break;

        case MMAL_PARAMETER_VIDEO_STABILISATION:
            status = set_camera_video_stabilisation(self->camera, (MMAL_BOOL_T)value);
            break;

        case MMAL_PARAMETER_EXPOSURE_MODE:
            status = set_camera_exposure_mode(self->camera, (MMAL_PARAM_EXPOSUREMODE_T)value);
            break;

        case MMAL_PARAMETER_AWB_MODE:
            status = set_camera_awb_mode(self->camera, (MMAL_PARAM_AWBMODE_T)value);
            break;

        case MMAL_PARAMETER_IMAGE_EFFECT:
            status = set_camera_image_fx(self->camera, (MMAL_PARAM_IMAGEFX_T)value);
            break;

        case MMAL_PARAMETER_FLASH:
            status = set_camera_flash_mode(self->camera, (MMAL_PARAMETER_CAMERA_INFO_FLASH_TYPE_T)value);
            break;

        case MMAL_PARAMETER_FLASH_SELECT:
            status = set_camera_flash_type(self->camera, (MMAL_PARAM_FLASH_T)value);
            break;

    }

    if (status != MMAL_SUCCESS){
        PyErr_SetString(PyExc_RuntimeError, (char *)mmal_status_to_string(status));
        return -1;
    }

    return 0;
}
PyObject *
RpiCamera_switch_output(RpiCamera *self, PyObject *args){

    uint8_t channel;
    MMAL_PORT_T *port;

    if (!PyArg_ParseTuple(args,"B", &channel))
        return NULL;

    if (channel < 0 || channel > 2){
        PyErr_Format(PyExc_ValueError, "Channel must be one of 0, 1, or 2 (given %d)", channel);
        return NULL;
    }

    //If this is the current port already, don't do anything.
    if(channel == self->output_port)
        Py_RETURN_NONE;

    port = self->camera->output[self->output_port];
    //Disable output port
    if(port && port->is_enabled){
        mmal_port_disable(port);
    }

    rpicamera_log_debug("Destroying old buffer pool...", NULL);
    mmal_port_pool_destroy(port, self->pool);
    self->pool = (MMAL_POOL_T *)NULL;
    

    rpicamera_log_debug("Switching to output %d", channel);
    self->output_port = channel;
    port = self->camera->output[channel];

    if (port->buffer_size < port->buffer_size_min)
        port->buffer_size = port->buffer_size_min;

    port->buffer_num = port->buffer_num_recommended;

    rpicamera_log_debug("Creating new pool...", NULL);
    self->pool = mmal_port_pool_create(port, port->buffer_num, port->buffer_size);

    if(!self->pool){
        PyErr_SetString(PyExc_RuntimeError, "Unable to create new buffer pool.");
        return NULL;
    }

    Py_RETURN_NONE;
}