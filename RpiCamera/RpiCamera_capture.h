#ifndef RPICAMERA_CAPTURE_H_
#define RPICAMERA_CAPTURE_H_

#include "RpiCamera_types.h"

int check_resize_image_buffer(RpiCamera *RpiCamera);
PyObject * RpiCamera_capture_stills(RpiCamera *RpiCamera, PyObject *arg, PyObject *kwd);
#endif