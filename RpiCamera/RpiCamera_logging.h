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

#ifndef RPICAMERA_LOGGER_H_
#define RPICAMERA_LOGGER_H_
#include <Python.h>
#include <pthread.h>

pthread_mutex_t logging_mutex;
extern PyObject *RPICAMERA_MODULE_LOGGER;
char RPICAMERA_MODULE_LOGGER_MSG[1024];

/*
    Convenience defines for logging at the five default levels.

    IMPORTANT:  Macros MUST CONTAIN AT LEAST TWO ARGUMENTS, THE C-STYLE
    STRING FORMAT AND AT LEAST ONE SUPLIMENTAL VALUE.

    IF THE LOG ENTRY IS A CONSTANT, NON-FORMATED STRING, ADD A TRAILING NULL:

        rpicamera_log_info("This is a constant string", NULL);
        rpicamera_log_info("This is a string with expansion of some value:  %d", 10);
*/
#define rpicamera_log_debug(format, ...) rpicamera_logger(10, format, __VA_ARGS__)
#define rpicamera_log_info(format, ...) rpicamera_logger(20, format, __VA_ARGS__)
#define rpicamera_log_warning(format, ...) rpicamera_logger(30, format, __VA_ARGS__)
#define rpicamera_log_error(format, ...) rpicamera_logger(40, format, __VA_ARGS__)
#define rpicamera_log_fatal(format, ...) rpicamera_logger(50, format, __VA_ARGS__)

int32_t rpicamera_logger(int32_t log_level, const char *format, ...);
#endif