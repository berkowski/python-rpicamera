/*
Copyright (c) Zachary Berkowitz
All rights reserved.

This file is part of the RpiCamera python extension for the
Raspberry Pi camera module.  Current sources can be found at
https://github.com/berkowski/python-rpicamera/

Substantial work was derived from James Hughes' Raspi* family of 
command-line driven programs, which can be found at:
https://github.com/raspberrypi/userland/

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

#ifndef RPICAMERA_CAMERA_SETTINGS_
#define RPICAMERA_CAMERA_SETTINGS_
#include "interface/mmal/mmal.h"

MMAL_STATUS_T get_camera_saturation(MMAL_COMPONENT_T *camera,  int32_t *value);
MMAL_STATUS_T get_camera_sharpness(MMAL_COMPONENT_T *camera, int32_t *value);
MMAL_STATUS_T get_camera_contrast(MMAL_COMPONENT_T *camera, int32_t *value);
MMAL_STATUS_T get_camera_brightness(MMAL_COMPONENT_T *camera, int32_t *value);
MMAL_STATUS_T get_camera_iso(MMAL_COMPONENT_T *camera, uint32_t *value);
MMAL_STATUS_T get_camera_metering_mode(MMAL_COMPONENT_T *camera, MMAL_PARAM_EXPOSUREMETERINGMODE_T *value);
MMAL_STATUS_T get_camera_exposure_compensation(MMAL_COMPONENT_T *camera, int32_t *value);
MMAL_STATUS_T get_camera_video_stabilisation(MMAL_COMPONENT_T *camera, MMAL_BOOL_T *value);
MMAL_STATUS_T get_camera_exposure_mode(MMAL_COMPONENT_T *camera, MMAL_PARAM_EXPOSUREMODE_T *value);
MMAL_STATUS_T get_camera_awb_mode(MMAL_COMPONENT_T *camera, MMAL_PARAM_AWBMODE_T *value);
MMAL_STATUS_T get_camera_image_fx(MMAL_COMPONENT_T *camera, MMAL_PARAM_IMAGEFX_T *value);
MMAL_STATUS_T get_camera_colour_fx(MMAL_COMPONENT_T *camera, MMAL_PARAMETER_COLOURFX_T *value);
MMAL_STATUS_T get_camera_rotation(MMAL_COMPONENT_T *camera, int32_t *value);
MMAL_STATUS_T get_camera_flips(MMAL_COMPONENT_T *camera, MMAL_PARAM_MIRROR_T *value);
// MMAL_STATUS_T get_camera_info(MMAL_COMPONENT_T *camera, MMAL_PARAMETER_CAMERA_INFO_T *value);
MMAL_STATUS_T get_camera_flash_mode(MMAL_COMPONENT_T *camera, MMAL_PARAM_FLASH_T *value);
MMAL_STATUS_T get_camera_flash_type(MMAL_COMPONENT_T *camera,  MMAL_PARAMETER_CAMERA_INFO_FLASH_TYPE_T *value);

MMAL_STATUS_T set_camera_saturation(MMAL_COMPONENT_T *camera,  int32_t value);
MMAL_STATUS_T set_camera_sharpness(MMAL_COMPONENT_T *camera, int32_t value);
MMAL_STATUS_T set_camera_contrast(MMAL_COMPONENT_T *camera, int32_t value);
MMAL_STATUS_T set_camera_brightness(MMAL_COMPONENT_T *camera, int32_t value);
MMAL_STATUS_T set_camera_iso(MMAL_COMPONENT_T *camera, uint32_t value);
MMAL_STATUS_T set_camera_metering_mode(MMAL_COMPONENT_T *camera, MMAL_PARAM_EXPOSUREMETERINGMODE_T value);
MMAL_STATUS_T set_camera_exposure_compensation(MMAL_COMPONENT_T *camera, int32_t value);
MMAL_STATUS_T set_camera_video_stabilisation(MMAL_COMPONENT_T *camera, MMAL_BOOL_T value);
MMAL_STATUS_T set_camera_exposure_mode(MMAL_COMPONENT_T *camera, MMAL_PARAM_EXPOSUREMODE_T value);
MMAL_STATUS_T set_camera_awb_mode(MMAL_COMPONENT_T *camera, MMAL_PARAM_AWBMODE_T value);
MMAL_STATUS_T set_camera_image_fx(MMAL_COMPONENT_T *camera, MMAL_PARAM_IMAGEFX_T value);
MMAL_STATUS_T set_camera_colour_fx(MMAL_COMPONENT_T *camera, const MMAL_PARAMETER_COLOURFX_T *value);
MMAL_STATUS_T set_camera_rotation(MMAL_COMPONENT_T *camera, int32_t value);
MMAL_STATUS_T set_camera_flips(MMAL_COMPONENT_T *camera, MMAL_PARAM_MIRROR_T value);
// MMAL_STATUS_T set_camera_info(MMAL_COMPONENT_T *camera, const MMAL_PARAMETER_CAMERA_INFO_T *value);
MMAL_STATUS_T set_camera_flash_mode(MMAL_COMPONENT_T *camera, MMAL_PARAM_FLASH_T value);
MMAL_STATUS_T set_camera_flash_type(MMAL_COMPONENT_T *camera,  MMAL_PARAMETER_CAMERA_INFO_FLASH_TYPE_T value);
#endif