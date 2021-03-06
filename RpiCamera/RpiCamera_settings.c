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
#include "RpiCamera_settings.h"
#include "interface/mmal/util/mmal_util_params.h"

MMAL_STATUS_T get_camera_saturation(MMAL_COMPONENT_T *camera,  int32_t *value){
	if (!camera)
		return MMAL_ENOTREADY;

	MMAL_RATIONAL_T ratio;
	MMAL_STATUS_T status;

	status = mmal_port_parameter_get_rational(camera->control, MMAL_PARAMETER_SATURATION, &ratio);
	*value = ratio.num;
	return status;
}

MMAL_STATUS_T get_camera_sharpness(MMAL_COMPONENT_T *camera, int32_t *value){
	if (!camera)
		return MMAL_ENOTREADY;

	MMAL_RATIONAL_T ratio;
	MMAL_STATUS_T status;

	status = mmal_port_parameter_get_rational(camera->control, MMAL_PARAMETER_SHARPNESS, &ratio);
	*value = ratio.num;
	return status;
}

MMAL_STATUS_T get_camera_contrast(MMAL_COMPONENT_T *camera, int32_t *value){
	if (!camera)
		return MMAL_ENOTREADY;

	MMAL_RATIONAL_T ratio;
	MMAL_STATUS_T status;

	status = mmal_port_parameter_get_rational(camera->control, MMAL_PARAMETER_CONTRAST, &ratio);
	*value = ratio.num;
	return status;
}

MMAL_STATUS_T get_camera_brightness(MMAL_COMPONENT_T *camera, int32_t *value){
	if (!camera)
		return MMAL_ENOTREADY;

	MMAL_RATIONAL_T ratio;
	MMAL_STATUS_T status;

	status = mmal_port_parameter_get_rational(camera->control, MMAL_PARAMETER_BRIGHTNESS, &ratio);
	*value = ratio.num;
	return status;
}

MMAL_STATUS_T get_camera_iso(MMAL_COMPONENT_T *camera, uint32_t *value){
	if (!camera)
		return MMAL_ENOTREADY;

	return mmal_port_parameter_get_uint32(camera->control, MMAL_PARAMETER_ISO, value);
}

MMAL_STATUS_T get_camera_metering_mode(MMAL_COMPONENT_T *camera, MMAL_PARAM_EXPOSUREMETERINGMODE_T *value){
	if (!camera)
		return MMAL_ENOTREADY;

	MMAL_PARAMETER_EXPOSUREMETERINGMODE_T param = {{MMAL_PARAMETER_EXP_METERING_MODE, sizeof(param)},
													0};
	MMAL_STATUS_T status;
	status =  mmal_port_parameter_get(camera->control, &param.hdr);
	*value = param.value;

	return status;
}

MMAL_STATUS_T get_camera_exposure_compensation(MMAL_COMPONENT_T *camera, int32_t *value){
	if (!camera)
		return MMAL_ENOTREADY;

	return mmal_port_parameter_get_int32(camera->control, MMAL_PARAMETER_EXPOSURE_COMP, value);
}

MMAL_STATUS_T get_camera_video_stabilisation(MMAL_COMPONENT_T *camera, MMAL_BOOL_T *value){
	if (!camera)
		return MMAL_ENOTREADY;

	return mmal_port_parameter_get_boolean(camera->control, MMAL_PARAMETER_VIDEO_STABILISATION, value);
}

MMAL_STATUS_T get_camera_exposure_mode(MMAL_COMPONENT_T *camera, MMAL_PARAM_EXPOSUREMODE_T *value){
	if (!camera)
		return MMAL_ENOTREADY;

	MMAL_PARAMETER_EXPOSUREMODE_T param = {{MMAL_PARAMETER_EXPOSURE_MODE, sizeof(param)},
											0};
	MMAL_STATUS_T  status;
	status =  mmal_port_parameter_get(camera->control, &param.hdr);
	*value = param.value;

	return status;
}

MMAL_STATUS_T get_camera_awb_mode(MMAL_COMPONENT_T *camera, MMAL_PARAM_AWBMODE_T *value){
	if (!camera)
		return MMAL_ENOTREADY;
	
	MMAL_PARAMETER_AWBMODE_T param = {{MMAL_PARAMETER_AWB_MODE, sizeof(param)},
									   0};
	MMAL_STATUS_T  status;
	status =  mmal_port_parameter_get(camera->control, &param.hdr);
	*value = param.value;

	return status;
}

MMAL_STATUS_T get_camera_image_fx(MMAL_COMPONENT_T *camera, MMAL_PARAM_IMAGEFX_T *value){
	if (!camera)
		return MMAL_ENOTREADY;

	MMAL_PARAMETER_IMAGEFX_T param = {{MMAL_PARAMETER_IMAGE_EFFECT, sizeof(param)},
									   0};
	MMAL_STATUS_T  status;
	status =  mmal_port_parameter_get(camera->control, &param.hdr);
	*value = param.value;

	return status;
}

MMAL_STATUS_T get_camera_colour_fx(MMAL_COMPONENT_T *camera, MMAL_PARAMETER_COLOURFX_T *value){
	if (!camera)
		return MMAL_ENOTREADY;
	
	return mmal_port_parameter_get(camera->control, &(value->hdr));
}

MMAL_STATUS_T get_camera_rotation(MMAL_COMPONENT_T *camera, int32_t *value){
	if (!camera)
		return MMAL_ENOTREADY;

	MMAL_STATUS_T status = MMAL_SUCCESS;
	uint8_t port = 0;
	for (port=0; port<3; port++){
		status |= mmal_port_parameter_get_int32(camera->output[port], MMAL_PARAMETER_ROTATION, value);
	}

	return status;
}

MMAL_STATUS_T get_camera_flips(MMAL_COMPONENT_T *camera, MMAL_PARAM_MIRROR_T *value){
	if (!camera)
		return MMAL_ENOTREADY;
	
	MMAL_PARAMETER_AWBMODE_T param = {{MMAL_PARAMETER_AWB_MODE, sizeof(param)},
									   0};
	MMAL_STATUS_T status = MMAL_SUCCESS;
	uint8_t port = 0;
	for (port=0; port<3; port++){
		status |=  mmal_port_parameter_get(camera->output[port], &param.hdr);
	}
	*value = param.value;

	return status;
}

MMAL_STATUS_T get_camera_flash_mode(MMAL_COMPONENT_T *camera, MMAL_PARAM_FLASH_T *value){
	if (!camera)
		return MMAL_ENOTREADY;

	MMAL_PARAMETER_FLASH_T param = {{MMAL_PARAMETER_FLASH, sizeof(param)},
											0};
	MMAL_STATUS_T  status;
	status =  mmal_port_parameter_get(camera->control, &param.hdr);
	*value = param.value;

	return status;
}

MMAL_STATUS_T get_camera_flash_type(MMAL_COMPONENT_T *camera,  MMAL_PARAMETER_CAMERA_INFO_FLASH_TYPE_T *value){
	if (!camera)
		return MMAL_ENOTREADY;

	MMAL_PARAMETER_FLASH_SELECT_T param = {{MMAL_PARAMETER_FLASH_SELECT , sizeof(param)},
											0};
	MMAL_STATUS_T  status;
	status =  mmal_port_parameter_get(camera->control, &param.hdr);
	*value = param.flash_type;

	return status;
}

MMAL_STATUS_T get_camera_info(MMAL_COMPONENT_T *camera, MMAL_PARAMETER_CAMERA_INFO_T *value){
	if (!camera)
		return MMAL_ENOTREADY;
	
	return mmal_port_parameter_get(camera->control, &(value->hdr));
}

MMAL_STATUS_T set_camera_saturation(MMAL_COMPONENT_T *camera,  int32_t value){
	if (!camera)
		return MMAL_ENOTREADY;
	
	if (value >= -100 && value <= 100){
		const MMAL_RATIONAL_T ratio = {value, 100};
		return mmal_port_parameter_set_rational(camera->control, MMAL_PARAMETER_SATURATION, ratio);
	}
	else{
		return MMAL_EINVAL;
	}

}

MMAL_STATUS_T set_camera_sharpness(MMAL_COMPONENT_T *camera, int32_t value){
	if (!camera)
		return MMAL_ENOTREADY;

	if (value >= -100 && value <= 100){
		const MMAL_RATIONAL_T ratio = {value, 100};
		return mmal_port_parameter_set_rational(camera->control, MMAL_PARAMETER_SHARPNESS, ratio);
	}
	else{
		return MMAL_EINVAL;
	}

}

MMAL_STATUS_T set_camera_contrast(MMAL_COMPONENT_T *camera, int32_t value){
	if (!camera)
		return MMAL_ENOTREADY;

	if (value >= -100 && value <= 100){
		const MMAL_RATIONAL_T ratio = {value, 100};
		return mmal_port_parameter_set_rational(camera->control, MMAL_PARAMETER_CONTRAST, ratio);
	}
	else{
		return MMAL_EINVAL;
	}
}

MMAL_STATUS_T set_camera_brightness(MMAL_COMPONENT_T *camera, int32_t value){
	if (!camera)
		return MMAL_ENOTREADY;

	if (value >= -100 && value <= 100){
		const MMAL_RATIONAL_T ratio = {value, 100};
		return mmal_port_parameter_set_rational(camera->control, MMAL_PARAMETER_BRIGHTNESS, ratio);
	}
	else{
		return MMAL_EINVAL;
	}
}

MMAL_STATUS_T set_camera_iso(MMAL_COMPONENT_T *camera, uint32_t value){
	if (!camera)
		return MMAL_ENOTREADY;

	return mmal_port_parameter_set_uint32(camera->control, MMAL_PARAMETER_ISO, value);
}

MMAL_STATUS_T set_camera_metering_mode(MMAL_COMPONENT_T *camera, MMAL_PARAM_EXPOSUREMETERINGMODE_T value){
	if (!camera)
		return MMAL_ENOTREADY;

	MMAL_PARAMETER_EXPOSUREMETERINGMODE_T param = {{MMAL_PARAMETER_EXP_METERING_MODE, sizeof(param)},
													value};

	return mmal_port_parameter_set(camera->control, &param.hdr);
}

MMAL_STATUS_T set_camera_exposure_compensation(MMAL_COMPONENT_T *camera, int32_t value){
	if (!camera)
		return MMAL_ENOTREADY;

	return mmal_port_parameter_set_int32(camera->control, MMAL_PARAMETER_EXPOSURE_COMP, value);
}

MMAL_STATUS_T set_camera_video_stabilisation(MMAL_COMPONENT_T *camera, MMAL_BOOL_T value){
	if (!camera)
		return MMAL_ENOTREADY;

	return mmal_port_parameter_set_boolean(camera->control, MMAL_PARAMETER_VIDEO_STABILISATION, value);

}

MMAL_STATUS_T set_camera_exposure_mode(MMAL_COMPONENT_T *camera, MMAL_PARAM_EXPOSUREMODE_T value){
	if (!camera)
		return MMAL_ENOTREADY;

	MMAL_PARAMETER_EXPOSUREMODE_T param = {{MMAL_PARAMETER_EXPOSURE_MODE, sizeof(param)},
										    value};

	return mmal_port_parameter_set(camera->control, &param.hdr);
}

MMAL_STATUS_T set_camera_awb_mode(MMAL_COMPONENT_T *camera, MMAL_PARAM_AWBMODE_T value){
	if (!camera)
		return MMAL_ENOTREADY;

	MMAL_PARAMETER_AWBMODE_T param = {{MMAL_PARAMETER_AWB_MODE, sizeof(param)},
									   value};

	return mmal_port_parameter_set(camera->control, &param.hdr);
}

MMAL_STATUS_T set_camera_image_fx(MMAL_COMPONENT_T *camera, MMAL_PARAM_IMAGEFX_T value){
	if (!camera)
		return MMAL_ENOTREADY;
	
	MMAL_PARAMETER_IMAGEFX_T param = {{MMAL_PARAMETER_IMAGE_EFFECT, sizeof(param)},
									   value};

	return mmal_port_parameter_set(camera->control, &param.hdr);
}

MMAL_STATUS_T set_camera_colour_fx(MMAL_COMPONENT_T *camera, const MMAL_PARAMETER_COLOURFX_T *value){
	if (!camera)
		return MMAL_ENOTREADY;

	// MMAL_PARAMETER_COLOURFX_T param = {{MMAL_PARAMETER_COLOUR_EFFECT, sizeof(param)},
	// 									0, 0, 0};

	// param.enable = value->enable;
	// param.u = value->u;
	// param.v = value->v;

	return mmal_port_parameter_set(camera->control, &(value->hdr));
}

MMAL_STATUS_T set_camera_rotation(MMAL_COMPONENT_T *camera, int32_t value){
	if (!camera)
		return MMAL_ENOTREADY;
	int32_t my_rotation = ((value % 360 ) / 90) * 90;
	MMAL_STATUS_T ret = MMAL_SUCCESS;
	uint8_t port = 0;
	for (port=0; port<3; port++){
		ret |= mmal_port_parameter_set_int32(camera->output[port], MMAL_PARAMETER_ROTATION, my_rotation);
	}

	return ret;

}

MMAL_STATUS_T set_camera_flips(MMAL_COMPONENT_T *camera, MMAL_PARAM_MIRROR_T value){
	if (!camera)
		return MMAL_ENOTREADY;

	MMAL_PARAMETER_MIRROR_T param = {{MMAL_PARAMETER_MIRROR, sizeof(param)},
									  value};

	MMAL_STATUS_T ret = MMAL_SUCCESS;
	uint8_t port = 0;
	for (port=0; port<3; port++){
		ret |= mmal_port_parameter_set(camera->output[port], &param.hdr);
	}

	return ret;
}

MMAL_STATUS_T set_camera_flash_mode(MMAL_COMPONENT_T *camera, MMAL_PARAM_FLASH_T value){
	if (!camera)
		return MMAL_ENOTREADY;

	MMAL_PARAMETER_FLASH_T param = {{MMAL_PARAMETER_FLASH, sizeof(param)},
											value};
	return  mmal_port_parameter_set(camera->control, &param.hdr);

}

MMAL_STATUS_T set_camera_flash_type(MMAL_COMPONENT_T *camera,  MMAL_PARAMETER_CAMERA_INFO_FLASH_TYPE_T value){
	if (!camera)
		return MMAL_ENOTREADY;

	MMAL_PARAMETER_FLASH_SELECT_T param = {{MMAL_PARAMETER_FLASH_SELECT, sizeof(param)},
											value};
	return mmal_port_parameter_set(camera->control, &param.hdr);

}
MMAL_STATUS_T set_camera_info(MMAL_COMPONENT_T *camera, const MMAL_PARAMETER_CAMERA_INFO_T *value){
	if (!camera)
		return MMAL_ENOTREADY;
	
	return mmal_port_parameter_set(camera->control, &(value->hdr));
}