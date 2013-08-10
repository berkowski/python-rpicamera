#ifndef RPICAMERA_TYPES_H_
#define RPICAMERA_TYPES_H_

#include "interface/mmal/mmal.h"
#include "interface/vcos/vcos.h"

typedef struct {
    PyObject_HEAD
    MMAL_COMPONENT_T *camera;
    MMAL_COMPONENT_T *encoder;
    MMAL_ES_FORMAT_T *format;
    MMAL_PORT_T *camera_preview_port;
    MMAL_PORT_T *camera_video_port;
    MMAL_PORT_T *camera_still_port;
    MMAL_PORT_T *output_port;
    
    VCOS_SEMAPHORE_T complete_semaphore;

    MMAL_POOL_T *pool;

    PyObject *image;
    char debug_flag;

} RpiCamera;


typedef struct {

    VCOS_SEMAPHORE_T complete_semaphore;
    MMAL_POOL_T *buffer_pool;
    uint8_t shift;
    uint8_t in_progress;
    uint8_t *image_data;
    uint32_t image_size;
    uint32_t bytes_written;
    uint8_t debug;

} STILL_CAPTURE_USERDATA_T;
#endif
