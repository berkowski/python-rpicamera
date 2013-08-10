#include "RpiCamera_logging.h"

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