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
#include "RpiCamera_logging.h"

/*  Simple logger hooks into the global PyObject RPICAMERA_MODULE_LOGGER defined
    in the module's startup function.  Mutexes around the shared string buffer
    and method calls prevent concurrent access from multiple threads.

    Should not be called directly, but instead through the 
    rpicamera_log_* family of macros defined in RpiCamera_logging.h

    Uses standard c string formating (a la sprintf, fprintf, etc.)
*/
int32_t rpicamera_logger(int32_t log_level, const char *format, ...){

    va_list args;
    int32_t status = 0;
    va_start(args, format);
    
    pthread_mutex_lock(&logging_mutex);
    status = vsprintf(RPICAMERA_MODULE_LOGGER_MSG, format, args);
    
    
    va_end(args);

    if (status < 0){
        PyErr_SetString(PyExc_ValueError, "Unable to format log message.");
        status = -1;
        goto finished;
    }


    if(PyObject_CallMethod(RPICAMERA_MODULE_LOGGER, "log", "is", log_level, RPICAMERA_MODULE_LOGGER_MSG)){
        status = 0;
    }
    else{
        status =-1;
    }
finished:
    pthread_mutex_unlock(&logging_mutex);
    return status;

}