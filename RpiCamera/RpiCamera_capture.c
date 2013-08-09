#include <Python.h>
// #define PY_ARRAY_UNIQUE_SYMBOL PyArray_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL RPICAMERA_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include "interface/vcos/vcos.h"
#include "interface/mmal/mmal.h"

#include "RpiCamera_types.h"
#include "RpiCamera_capture.h"
#include "RpiCamera_logging.h"

static void camera_output_callback(MMAL_PORT_T *port, MMAL_BUFFER_HEADER_T *buffer)
{
	rpicamera_log_debug("Starting capture callback.", NULL);
   	STILL_CAPTURE_USERDATA_T *pData = (STILL_CAPTURE_USERDATA_T *)port->userdata;

	int complete = 0;
	int bytes_to_write = 0;
	uint8_t *img_buffer, *data_buffer;

	uint32_t k;

	// We pass our file handle and other stuff in via the userdata field.


   	if (pData){
      	bytes_to_write = buffer->length - buffer->offset;

     	if (pData->bytes_written + bytes_to_write > pData->image_size){
     		rpicamera_log_warning("Possible overflow condition.", NULL);
     		rpicamera_log_warning("MAXSIZE:  %d, BYTES_WRITTEN:  %d", pData->image_size, pData->bytes_written);
     		rpicamera_log_warning("BYTES_WRITTEN:  %d", pData->bytes_written);
     		rpicamera_log_warning("BUFFER_OFFSET:  %d", buffer->offset);
     		rpicamera_log_warning("BUFFER_LENGTH:  %d", buffer->length);
     		bytes_to_write = pData->image_size - pData->bytes_written;
     		complete = 1;
     	}

      	if (bytes_to_write){
      		if (pData->debug)
      			rpicamera_log_debug("Locking buffer.", NULL);

	       	mmal_buffer_header_mem_lock(buffer);
	       	
	       	if (pData->debug)
				rpicamera_log_debug("writing %d bytes of data.", bytes_to_write);	       		

			img_buffer = pData->image_data + pData->bytes_written;
			data_buffer = buffer->data + buffer->offset;

			for(k=0; k<bytes_to_write; k++){
				*img_buffer += (*data_buffer >> pData->shift);
				img_buffer++;
				data_buffer++;
			}

	       	pData->bytes_written += bytes_to_write;		
			
			mmal_buffer_header_mem_unlock(buffer);
			if (pData->debug)
				rpicamera_log_debug("Removing buffer lock.", NULL);

      	}

		// Check end of frame or error
      	if (buffer->flags & (MMAL_BUFFER_HEADER_FLAG_FRAME_END | MMAL_BUFFER_HEADER_FLAG_TRANSMISSION_FAILED))
        	complete = 1;
   	}
   	
   	else{
    
      	rpicamera_log_warning("Received a camera still buffer callback with no state", NULL);
   	
   	}

	// release buffer back to the pool
	mmal_buffer_header_release(buffer);

	// and send one back to the port (if still open)
	if (port->is_enabled){
		MMAL_STATUS_T status;
      	MMAL_BUFFER_HEADER_T *new_buffer = mmal_queue_get(pData->buffer_pool->queue);

      	// and back to the port from there.
      	if (new_buffer){
         	status = mmal_port_send_buffer(port, new_buffer);
      	}

      	if (!new_buffer || status != MMAL_SUCCESS)
         	rpicamera_log_warning("Unable to return the buffer to the camera still port", NULL);
   	}

   	if (complete){
      	vcos_semaphore_post(&(pData->complete_semaphore));
   	}
}


PyObject *RpiCamera_capture_stills(RpiCamera *self, PyObject *arg, PyObject *kwd){

	static char *kwlist[] = {"frames", "shift", NULL};

	MMAL_ES_FORMAT_T *format = self->output_port->format;
	MMAL_BUFFER_HEADER_T *pool_buffer;

	STILL_CAPTURE_USERDATA_T callback_data;
   	VCOS_STATUS_T vcos_status = VCOS_SUCCESS;

	uint32_t status = 0;
	uint32_t num_frames = 1;
	uint8_t  shift = 0;
	//Indexes
	uint32_t k = 0;

	if (!PyArg_ParseTupleAndKeywords(arg, kwd, "|IB", kwlist, &num_frames, &shift))
		return NULL;

	if (shift > 7){
		PyErr_Format(PyExc_ValueError, "shift value must be less than 8, provided %d", shift);
		return NULL;
	}

	if (check_resize_image_buffer(self))
		//There was a problem checking/resizing the image buffer.
		//Errors have already been raised, so just exit.
		return NULL;

	//Clear image object
	PyArray_FILLWBYTE((PyArrayObject *)self->image, 0);


	//Setup callback data structure
	callback_data.buffer_pool = self->pool;
	callback_data.bytes_written = 0;
	callback_data.shift = shift;

	callback_data.image_size = (uint32_t) PyArray_Size(self->image);
	callback_data.image_data = (uint8_t *) PyArray_GETPTR1((PyArrayObject *)self->image, 0);

	vcos_status = vcos_semaphore_create(&callback_data.complete_semaphore, "RpiCamera-sem", 0);
	vcos_assert(vcos_status == VCOS_SUCCESS);

	self->output_port->userdata = (struct MMAL_PORT_USERDATA_T *)&callback_data;
	status = mmal_port_enable(self->output_port, camera_output_callback);
	if (status != MMAL_SUCCESS){
		PyErr_SetString(PyExc_RuntimeError, "Could not enable output port");
		return NULL;
	}

	//Send all buffers to the output port
	rpicamera_log_debug("Sending %d buffers...\n", mmal_queue_length(self->pool->queue));
    for (k=0;k<mmal_queue_length(self->pool->queue); k++){
 		pool_buffer = mmal_queue_get(self->pool->queue);
 		if (!pool_buffer)
            rpicamera_log_warning("Unable to get a required buffer %d from pool queue", k);

        if (mmal_port_send_buffer(self->output_port, pool_buffer)!= MMAL_SUCCESS)
            rpicamera_log_warning("Unable to send a buffer to camera output port (%d)", k);
                 
 	}

 	rpicamera_log_info("Starting capture.", NULL);
	// Fire the capture
	for(k=0;k<num_frames;k++){
		status = mmal_port_parameter_set_boolean(self->output_port, MMAL_PARAMETER_CAPTURE, MMAL_TRUE);

		if(status != MMAL_SUCCESS){
			rpicamera_log_error("%s: Failed to start capture", __func__);
		}
		else{
			// Wait for capture to complete
			// For some reason using vcos_semaphore_wait_timeout sometimes returns immediately with bad parameter error
			// even though it appears to be all correct, so reverting to untimed one until figure out why its erratic
			vcos_semaphore_wait(&callback_data.complete_semaphore);
			rpicamera_log_info("Frame %d finished.", k+1);			
		}

		callback_data.bytes_written = 0;

	}
   	
		
	//Delete the semiphore
	vcos_semaphore_delete(&callback_data.complete_semaphore);


	//Disable the output port
	if(self->output_port && self->output_port->is_enabled)
		mmal_port_disable(self->output_port);


	Py_RETURN_NONE;
}


int check_resize_image_buffer(RpiCamera *RpiCamera){

	MMAL_ES_FORMAT_T *format = RpiCamera->output_port->format;
	PyArrayObject *image_array = (PyArrayObject *)RpiCamera->image;
	PyObject *new_array=NULL;


	int32_t format_size = format->es->video.crop.height * format->es->video.crop.width;
	const int32_t  image_size = PyArray_Size(RpiCamera->image);

	uint8_t require_resize = 0;

	switch(format->encoding){
		// 24bit RGB output 3 channels, 8bit samples
		case MMAL_ENCODING_BGR24:
			format_size *= 3;
			break;

		// YUV420, 12 effective bits per pixel (H*W Y values, (H*W/4) U and V)
		case MMAL_ENCODING_I420:
			format_size *= 1.5;
			break;
		
		default:
			PyErr_Format(PyExc_ValueError, "Unknown encoding type id:  %d", format->encoding);
			return -1;
	}



	if(!image_array){
		rpicamera_log_warning("Null image buffer (should not happen), forcing resize.\n", NULL);
		require_resize = 1;
	}
	else if (format_size != image_size){
		rpicamera_log_debug("Required buffer size has changed:  old:  %d, new:  %d\n", image_size, format_size);
		require_resize = 1;
	}


	if (require_resize != 0){
		rpicamera_log_debug("Resizing buffer to length %d", format_size);
		new_array = PyArray_FromDims(1, (int *)&format_size, NPY_UINT8);

		if (!new_array){
			return -1;
		}

		Py_INCREF(new_array);
		RpiCamera->image = new_array;
		
		Py_XDECREF(image_array);
		
	}

	return 0;
}