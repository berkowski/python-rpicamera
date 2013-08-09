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