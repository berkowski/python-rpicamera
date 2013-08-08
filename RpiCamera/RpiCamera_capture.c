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

static void camera_output_callback(MMAL_PORT_T *port, MMAL_BUFFER_HEADER_T *buffer)
{
   int complete = 0;
   int buffer_length = 0;
   int bytes_to_write = 0;

   // We pass our file handle and other stuff in via the userdata field.

   	fprintf(stderr, "Starting capture callback\n");
   	STILL_CAPTURE_USERDATA_T *pData = (STILL_CAPTURE_USERDATA_T *)port->userdata;

   	if (pData){
      	buffer_length = buffer->length;

     	if (pData->bytes_written + buffer_length > pData->image_size){
     		fprintf(stderr, "WARNING:  PREVEINTING BUFFER OVERFLOW\n");
     		fprintf(stderr, "MAXSIZE:  %d\n", pData->image_size);
     		fprintf(stderr, "BYTES_WRITTEN:  %d\n", pData->bytes_written);
     		fprintf(stderr, "BUFFER_OFFSET:  %d\n", buffer->offset);
     		fprintf(stderr, "BUFFER_LENGTH:  %d\n", buffer_length);
     		buffer_length = pData->image_size - pData->bytes_written;
     		complete = 1;
     	}

      	if (buffer_length){
      		fprintf(stderr, "Locking buffer\n");
	       	mmal_buffer_header_mem_lock(buffer);
	       	fprintf(stderr, "writing %d bytes of data\n", buffer_length);
	       	memcpy(pData->image_data, buffer->data, buffer_length);
	       	pData->bytes_written += buffer_length;
			
			mmal_buffer_header_mem_unlock(buffer);
      	}

		// Check end of frame or error
      	if (buffer->flags & (MMAL_BUFFER_HEADER_FLAG_FRAME_END | MMAL_BUFFER_HEADER_FLAG_TRANSMISSION_FAILED))
        	complete = 1;
   	}
   	
   	else{
    
      	fprintf(stderr, "Received a camera still buffer callback with no state");
   	
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
         	fprintf(stderr, "Unable to return the buffer to the camera still port");
   	}

   	if (complete){
      	vcos_semaphore_post(&(pData->complete_semaphore));
   	}
}


PyObject *RpiCamera_capture_stills(RpiCamera *self, PyObject *arg, PyObject *kwd){


	MMAL_ES_FORMAT_T *format = self->output_port->format;
	MMAL_BUFFER_HEADER_T *pool_buffer;

	PyArrayObject *img = (PyArrayObject *)self->image;
	npy_intp img_size = PyArray_Size(self->image);
	fprintf(stderr, "IMGSIZE:  %d\n", (uint32_t)img_size);

	STILL_CAPTURE_USERDATA_T callback_data;
   	VCOS_STATUS_T vcos_status = VCOS_SUCCESS;

	uint32_t status = 0;

	//Indexes
	uint32_t k = 0;

	if (check_resize_image_buffer(self))
		//There was a problem checking/resizing the image buffer.
		//Errors have already been raised, so just exit.
		return NULL;

	//Clear image object
	PyArray_FILLWBYTE(img, 0);


	//Setup callback data structure
	callback_data.buffer_pool = self->pool;
	callback_data.bytes_written = 0;


	callback_data.image_size = (uint32_t) img_size;
	callback_data.image_data = (uint8_t *) PyArray_GETPTR1(img, 0);

	vcos_status = vcos_semaphore_create(&callback_data.complete_semaphore, "RpiCamera-sem", 0);
	vcos_assert(vcos_status == VCOS_SUCCESS);

	self->output_port->userdata = (struct MMAL_PORT_USERDATA_T *)&callback_data;
	status = mmal_port_enable(self->output_port, camera_output_callback);
	if (status != MMAL_SUCCESS){
		PyErr_SetString(PyExc_RuntimeError, "Could not enable output port");
		return NULL;
	}

	//Send all buffers to the output port
	fprintf(stderr, "Sending %d buffers...\n", mmal_queue_length(self->pool->queue));
    for (k=0;k<mmal_queue_length(self->pool->queue); k++){
 		pool_buffer = mmal_queue_get(self->pool->queue);
 		if (!pool_buffer)
            fprintf(stderr, "Unable to get a required buffer %d from pool queue", k);

        if (mmal_port_send_buffer(self->output_port, pool_buffer)!= MMAL_SUCCESS)
            fprintf(stderr, "Unable to send a buffer to camera output port (%d)", k);
                 
 	}

 	fprintf(stderr, "Starting capture\n");
	// Fire the capture
   	if (mmal_port_parameter_set_boolean(self->output_port, MMAL_PARAMETER_CAPTURE, MMAL_TRUE) != MMAL_SUCCESS){
		fprintf(stderr, "%s: Failed to start capture", __func__);
    }
    else {
		// Wait for capture to complete
		// For some reason using vcos_semaphore_wait_timeout sometimes returns immediately with bad parameter error
		// even though it appears to be all correct, so reverting to untimed one until figure out why its erratic
		vcos_semaphore_wait(&callback_data.complete_semaphore);

		fprintf(stderr, "Finished capture %d\n", 1);

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
	PyArrayObject *new_array=NULL;


	int32_t format_shape[2];
	npy_intp *image_shape = NULL;
	int new_image_shape;

	int32_t  format_dim, image_dim;
	char require_resize = 0;
	uint8_t dim;

	switch(format->encoding){
		// 24bit RGB output 
		case MMAL_ENCODING_BGR24:
			format_dim = 3;
			break;

		case MMAL_ENCODING_I420:
			format_dim = 2;
			break;
		
		default:
			PyErr_Format(PyExc_ValueError, "Unknown encoding type id:  %d", format->encoding);
			return -1;
	}

	format_shape[0] = format->es->video.crop.height;
	format_shape[1] = format->es->video.crop.width;

	if(!image_array){
		require_resize = 1;
	}
	else {

		image_dim = PyArray_NDIM(image_array);
		image_shape = PyArray_DIMS(image_array);

		if (image_dim != format_dim)
			require_resize = 1;
		
		else {
			for(dim=0; dim<image_dim; dim++){
				if (image_shape[dim] != format_shape[dim]){
					require_resize = 1;
					break;
				}
			}
		}

	}

	if (require_resize != 0){

		switch(format->encoding){
			case MMAL_ENCODING_BGR24:
				new_image_shape = 3 * format_shape[0] * format_shape[1];
				break;

			case MMAL_ENCODING_I420:
				new_image_shape = format_shape[0] * format_shape[1] * 1.5;
				// new_image_shape[2] = 0;
				break;
			
			default:
				PyErr_Format(PyExc_RuntimeError, "THIS IS A BUG; should be caught earlier.");
				return -1;
		}

		new_array = PyArray_FromDims(1, &new_image_shape, NPY_UINT8);

		if (!new_array){
			PyErr_SetString(PyExc_RuntimeError, "Problem creating new array...");
			return -1;
		}
		RpiCamera->image = (PyObject *)new_array; //
		Py_INCREF(RpiCamera->image);
		
		if(image_array){
			fprintf(stderr, "Decref old array\n");
			Py_DECREF(image_array);
		}

	}
	return 0;
}