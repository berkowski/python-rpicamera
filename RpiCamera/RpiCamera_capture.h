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

#ifndef RPICAMERA_CAPTURE_H_
#define RPICAMERA_CAPTURE_H_

#include "RpiCamera_types.h"

PyDoc_STRVAR(capture_still_image__doc__,
"capture_still_image([channel=RpiCamera.CAMERA_STILL])\n\n"
"Captures a still image using the selected output channel.\n"
"The captured image is stored in the read-only buffer attribute\n"
"Camera.image\n\n"
":param channel:  Output channel to use.\n"
":type channel: int\n\n"
"Camera channels:\n"
"   Use following constants to set the output channel:\n"
"   RpiCamera.CAMERA_STILL      Use the still capture output.\n"
"   RpiCamera.CAMERA_PREVIEW    Use the preview output."
);
PyObject *RpiCamera_capture_still(RpiCamera *self, PyObject *args, PyObject *kwds);

PyDoc_STRVAR(integrate_preview_frames__doc__,
"integrate_preview_frames([frames=1, shift=0])\n\n"
"Sums a series of frames captured on the camera's preview channel.\n"
"The captured image is stored in the read-only buffer attribute\n"
"Camera.image\n\n"
":param frames:  Number for consecutive frames to capture.\n"
":type frames:  int.\n\n"
":param shift:  right-bitshift to apply before summing each frame.\n"
":type shift:  int.\n\n\n"
"No overflow checks are performed while integrating, so care should be\n"
"taken to choose an appropriate value for :param shift:.  Values for :param shift:\n"
"essentially divide the current pixel value by 2**(shift) before summation.  Shift\n"
"values > 7 are not allowed as they would result in identical pixels of 0 regardless\n"
"of intput value (pixels are 8-bit unsigned ints)");
PyObject *RpiCamera_integrated_preview(RpiCamera *RpiCamera, PyObject *arg, PyObject *kwd);
#endif