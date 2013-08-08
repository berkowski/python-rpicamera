#include "RpiCamera_logging.h"

int32_t rpicamera_logger(int32_t log_level, const char *format, ...){

	va_list args;
	int32_t status = 0;
	va_start(args, format);
	status = vsprintf(RPICAMERA_MODULE_LOGGER_MSG, format, args);
	va_end(args);

	if (status < 0){
		PyErr_SetString(PyExc_ValueError, "Unable to format log message.");
		return -1;
	}


	if(PyObject_CallMethod(RPICAMERA_MODULE_LOGGER, "log", "is", log_level, RPICAMERA_MODULE_LOGGER_MSG)){
		return 0;
	}
	else{
		return -1;
	}
}