#ifndef RPICAMERA_LOGGER_H_
#define RPICAMERA_LOGGER_H_
#include <Python.h>

extern PyObject *RPICAMERA_MODULE_LOGGER;
char RPICAMERA_MODULE_LOGGER_MSG[1024];

// #define rpicamera_log_debug(...) sprintf(RPICAMERA_MODULE_LOGGER_MSG, __VA_ARGS__); PyObject_CallMethod(RPICAMERA_MODULE_LOGGER, "debug", "N", PyUnicode_FromString(RPICAMERA_MODULE_LOGGER_MSG))
// #define rpicamera_log_info(...) sprintf(RPICAMERA_MODULE_LOGGER_MSG, __VA_ARGS__); PyObject_CallMethod(RPICAMERA_MODULE_LOGGER, "info", "N", PyUnicode_FromString(RPICAMERA_MODULE_LOGGER_MSG))
// #define rpicamera_log_warning(...) PyObject_CallMethod(RPICAMERA_MODULE_LOGGER, "warning", "N", PyUnicode_FromFormat(__VA_ARGS__))
// #define rpicamera_log_error(...) PyObject_CallMethod(RPICAMERA_MODULE_LOGGER, "error", "N", PyUnicode_FromFormat(__VA_ARGS__))
// #define rpicamera_log_fatal(...) PyObject_CallMethod(RPICAMERA_MODULE_LOGGER, "fatal", "N", PyUnicode_FromFormat(__VA_ARGS__))

#define rpicamera_log_debug(format, ...) rpicamera_logger(10, format, __VA_ARGS__)
#define rpicamera_log_info(format, ...) rpicamera_logger(20, format, __VA_ARGS__)
#define rpicamera_log_warning(format, ...) rpicamera_logger(30, format, __VA_ARGS__)
#define rpicamera_log_error(format, ...) rpicamera_logger(40, format, __VA_ARGS__)
#define rpicamera_log_fatal(format, ...) rpicamera_logger(50, format, __VA_ARGS__)

int32_t rpicamera_logger(int32_t log_level, const char *format, ...);
#endif