#include "interface/mmal/util/mmal_default_components.h"
#include "interface/mmal/util/mmal_util_params.h"

#include "RpiCamera.h"
#include "camera_settings.h"

// Standard port setting for the camera component
#define MMAL_CAMERA_PREVIEW_PORT 0
#define MMAL_CAMERA_VIDEO_PORT 1
#define MMAL_CAMERA_CAPTURE_PORT 2

// Stills format information
#define DEFAULT_STILLS_FRAME_RATE_NUM 3
#define DEFAULT_STILLS_FRAME_RATE_DEN 1
#define DEFAULT_STILLS_WIDTH 2592
#define DEFAULT_STILLS_HEIGHT 1944

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
        // vcos_log_error("Received unexpected camera control callback event, 0x%08x", buffer->cmd);
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

MMAL_STATUS_T __setup_camera(Camera *CameraObj){

    MMAL_STATUS_T status;
    MMAL_COMPONENT_T *camera = NULL;
    MMAL_ES_FORMAT_T *format;
    MMAL_PORT_T *preview_port, *video_port, *still_port;

    if (!CameraObj)
        // Camera object needs to be initialized first
        goto error;

    camera = CameraObj->camera;

    status = mmal_component_create(MMAL_COMPONENT_DEFAULT_CAMERA, &camera);

    if (status != MMAL_SUCCESS)
    {
      // vcos_log_error("Failed to create camera component");
      goto error;
    }

    if (!camera->output_num)
    {
      // vcos_log_error("Camera doesn't have output ports");
      goto error;
    }

    // Enable the camera, and tell it its control callback function
    status = mmal_port_enable(camera->control, __camera_control_callback);

    if (status)
    {
      // vcos_log_error("Unable to enable control port : error %d", status);
      goto error;
    }

    //  set up the camera configuration
    {
      MMAL_PARAMETER_CAMERA_CONFIG_T cam_config =
        {
            {MMAL_PARAMETER_CAMERA_CONFIG, sizeof(cam_config)},
            .max_stills_w = DEFAULT_STILLS_WIDTH,
            .max_stills_h = DEFAULT_STILLS_HEIGHT,
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
        // vcos_log_error("Unable to set camera config");
        goto error;
    }

    status = __set_default_camera_parameters(camera);

    if (status){
        // vcos_log_error("Unable to set default camera parameters");
        goto error;
    }

    preview_port = camera->output[MMAL_CAMERA_PREVIEW_PORT];
    format = preview_port->format;

    format->encoding = MMAL_ENCODING_OPAQUE;
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
      // vcos_log_error("camera viewfinder format couldn't be set");
      goto error;
    }

    video_port = camera->output[MMAL_CAMERA_VIDEO_PORT];
    mmal_format_full_copy(video_port->format, format);

    status = mmal_port_format_commit(video_port);

    if (status){
      // vcos_log_error("camera video format couldn't be set");
      goto error;
    }

    // Ensure there are enough buffers to avoid dropping frames
    if (video_port->buffer_num < DEFAULT_VIDEO_OUTPUT_BUFFERS_NUM)
      video_port->buffer_num = DEFAULT_VIDEO_OUTPUT_BUFFERS_NUM;

    still_port = camera->output[MMAL_CAMERA_CAPTURE_PORT];
    format = still_port->format;

    format->encoding = MMAL_ENCODING_I420;
    format->encoding_variant = MMAL_ENCODING_I420;
    
    format->es->video.width = DEFAULT_STILLS_WIDTH;
    format->es->video.height = DEFAULT_STILLS_HEIGHT;
    format->es->video.crop.x = 0;
    format->es->video.crop.y = 0;
    format->es->video.crop.width = DEFAULT_STILLS_WIDTH;
    format->es->video.crop.height = DEFAULT_STILLS_HEIGHT;
    format->es->video.frame_rate.num = DEFAULT_STILLS_FRAME_RATE_NUM;
    format->es->video.frame_rate.den = DEFAULT_STILLS_FRAME_RATE_DEN;

    if (still_port->buffer_size < still_port->buffer_size_min)
      still_port->buffer_size = still_port->buffer_size_min;

    still_port->buffer_num = still_port->buffer_num_recommended;

    status = mmal_port_format_commit(still_port);

    if (status)
    {
      // vcos_log_error("camera still format couldn't be set");
      goto error;
    }

    /* Enable component */
    status = mmal_component_enable(camera);

    if (status)
    {
      // vcos_log_error("camera component couldn't be enabled");
      goto error;
    }
    
    CameraObj->camera = camera;

    return status;

error:

    if (camera)
        mmal_component_destroy(camera);

    return status;

}

// New, Init, dealloc prototypes
static PyObject *
Camera_new(PyTypeObject *type, PyObject *args, PyObject *kwds){

    Camera *self;
    npy_intp img_dims[3] = {0, 0, 0};

    self = (Camera *)type->tp_alloc(type, 0);
    if (self == NULL)
        return (PyObject *)self;
    
    self->image = PyArray_SimpleNew(2, &img_dims, NPY_USHORT);
    Py_INCREF(self->image);

    //Take ownership & attach a logger
/*
    PyObject *logging_module = PyImport_AddModule("logging");
    PyObject *logger = NULL;

    if(logging_module != NULL){
        logger = PyObject_CallMethodObjArgs(logging_module, PyUnicode_FromString("getLogger"),
            PyUnicode_FromString("RpiCamera"), NULL);

        PyObject_CallMethod(logger, PyUnicode_FromString("setLevel"), 
            PyUnicode_FromString("INFO"), NULL);

        Py_INCREF(logger);
    }
    
    self->logger = logger;
*/
    

    return (PyObject *)self;

}



static int 
Camera_init(Camera *self, PyObject *args, PyObject *kwds){

    MMAL_STATUS_T status;
    MMAL_ES_FORMAT_T *format;

    status = __setup_camera(self);

    if (status != MMAL_SUCCESS)
        return (int)status;

    // self->camera_preview_port = self->camera->output[MMAL_CAMERA_PREVIEW_PORT];
    // self->camera_video_port = self->camera->output[MMAL_CAMERA_VIDEO_PORT];
    // self->camera_still_port = self->camera->output[MMAL_CAMERA_CAPTURE_PORT];

    return 0;
}

static void 
Camera_dealloc (Camera *self){

    Py_XDECREF(self->image);

    if (self->camera)
        mmal_component_destroy(self->camera);

    Py_TYPE(self)->tp_free((PyObject *)self);
}

/*  Getters & Setters

*/
static PyObject *
Camera_get_image(Camera *self, void *closure){

    Py_INCREF(self->image);
    return self->image;
}

static PyObject *
Camera_get_camera_setting(Camera *self, void *closure){

    int32_t param = (int32_t)closure;
    int32_t value;
    PyObject *result;

    MMAL_STATUS_T status;

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

    }

    if (status != MMAL_SUCCESS){
        //do some error handling
        return NULL;
    }

    Py_INCREF(result);
    return result;
}

static int
Camera_set_camera_setting(Camera *self, PyObject *py_value, void *closure){

    int32_t param = (int32_t)closure;
    int32_t value;
    PyObject *result;

    MMAL_STATUS_T status;

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

    }

    if (status != MMAL_SUCCESS){
        PyErr_SetString(PyExc_RuntimeError, mmal_status_to_string(status));
        return -1;
    }

    return 0;
}
static PyObject *
Camera_get_camera_settings(Camera *self, void *closure){

    PyObject *settings = PyDict_New();
    if (!settings)
        return settings;

    int32_t  p32;
    MMAL_STATUS_T status = MMAL_SUCCESS;


    status |= get_camera_saturation(self->camera,  &p32);
    PyDict_SetItemString(settings, "saturation", PyLong_FromLong((long)p32));

    status |= get_camera_sharpness(self->camera, &p32);
    PyDict_SetItemString(settings, "sharpness", PyLong_FromLong((long)p32));

    status |= get_camera_contrast(self->camera, &p32);
    PyDict_SetItemString(settings, "contrast", PyLong_FromLong((long)p32));

    status |= get_camera_brightness(self->camera, &p32);
    PyDict_SetItemString(settings, "brightness", PyLong_FromLong((long)p32));

    status |= get_camera_iso(self->camera, (uint32_t *) &p32);
    PyDict_SetItemString(settings, "iso", PyLong_FromLong((long)p32));

    status |= get_camera_metering_mode(self->camera, (MMAL_PARAM_EXPOSUREMETERINGMODE_T *) &p32);
    PyDict_SetItemString(settings, "metering_mode", PyLong_FromLong((long)p32));

    status |= get_camera_exposure_compensation(self->camera, &p32);
    PyDict_SetItemString(settings, "exposure_compensation", PyLong_FromLong((long)p32));

    status |= get_camera_video_stabilisation(self->camera, (MMAL_BOOL_T *) &p32);
    PyDict_SetItemString(settings, "video_stabilisation", PyLong_FromLong((long)p32));

    status |= get_camera_exposure_mode(self->camera, (MMAL_PARAM_EXPOSUREMODE_T *) &p32);
    PyDict_SetItemString(settings, "exposure_mode", PyLong_FromLong((long)p32));

    status |= get_camera_awb_mode(self->camera, (MMAL_PARAM_AWBMODE_T *) &p32);
    PyDict_SetItemString(settings, "awb_mode", PyLong_FromLong((long)p32));

    status |= get_camera_image_fx(self->camera, (MMAL_PARAM_IMAGEFX_T *) &p32);
    PyDict_SetItemString(settings, "image_fx", PyLong_FromLong((long)p32));

    // status |= get_camera_colour_fx(self->camera, (MMAL_PARAMETER_COLOURFX_T *) &p32);
    // PyDict_SetItemString(settings, "", PyLong_FromLong((long)p32));

    status |= get_camera_rotation(self->camera, &p32);
    PyDict_SetItemString(settings, "rotation", PyLong_FromLong((long)p32));

    status |= get_camera_flips(self->camera, (MMAL_PARAM_MIRROR_T *) &p32);
    PyDict_SetItemString(settings, "flips", PyLong_FromLong((long)p32));


    return settings;
}