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

#ifndef RPICAMERA_TYPES_H_
#define RPICAMERA_TYPES_H_

#include "interface/mmal/mmal.h"
#include "interface/vcos/vcos.h"

// Standard port setting for the camera component
#define MMAL_CAMERA_PREVIEW_PORT 0
#define MMAL_CAMERA_VIDEO_PORT 1
#define MMAL_CAMERA_CAPTURE_PORT 2

typedef struct {
    PyObject_HEAD
    MMAL_COMPONENT_T *camera;           /* Holds the camera component */
    MMAL_COMPONENT_T *encoder;          /* Holds the encoder component (NOT USED) */
    // MMAL_ES_FORMAT_T *format;           
    // MMAL_PORT_T *camera_preview_port;   
    // MMAL_PORT_T *camera_video_port;
    // MMAL_PORT_T *camera_still_port;
    uint8_t output_port;                     /* Current port id we're taking stills from */
    VCOS_SEMAPHORE_T *complete_semaphore;    /* Our processing semaphore */

    MMAL_POOL_T *pool;                      /* Our processing pool */

    PyObject *image;                        /* Pointer to the 1-D Numpy array holding the last still image*/
    char debug_flag;                        /* Flag for extra verbose logging */

} RpiCamera;


typedef struct {

    VCOS_SEMAPHORE_T *complete_semaphore;    /* Pointer to the processing semaphore of the parent camera object */
    MMAL_POOL_T *buffer_pool;               /* Poitner to the processing pool of the parent camera object */
    uint8_t shift;                          /* Right-shift level for multi-frame integration.*/
    uint8_t *image_data;                    /* Pointer to the memory space of the camera's 'image' member.*/
    uint32_t image_size;                    /* Size of the memory buffer in bytes*/
    uint32_t bytes_written;                 /* How many bytes we've already written to the buffer */
    uint8_t debug;                          /* Enable/disable verbose debug logging (may slow down performance)*/
    uint32_t num_frames;
    uint32_t current_frame;
    uint32_t capture_complete;

} INTEGRATED_PREVIEW_USERDATA_T;
#endif
