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

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL RPICAMERA_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include "interface/vcos/vcos.h"
#include "interface/mmal/mmal.h"
#include "interface/mmal/util/mmal_util_params.h"

#include "RpiCamera_types.h"
#include "RpiCamera_capture.h"
#include "RpiCamera_logging.h"

extern PyObject *RpiCamera_switch_output(RpiCamera *self, PyObject *args);

void integrated_preview_callback(MMAL_PORT_T *port, MMAL_BUFFER_HEADER_T *buffer)
{
   	INTEGRATED_PREVIEW_USERDATA_T *pData = (INTEGRATED_PREVIEW_USERDATA_T *)port->userdata;

   	int release_semaphore = 0;
	int bytes_to_write = 0;
	uint8_t *img_buffer, *data_buffer;
	// uint8_t 
	// uint8_t capture_frame = 1;
	uint32_t k;

	// We pass our file handle and other stuff in via the userdata field.
	bytes_to_write = buffer->length - buffer->offset;

	if (!pData){
		//No callballback data associated, do nothing.
		rpicamera_log_warning("Received a camera still buffer callback with no state", NULL);
	}

	else if (buffer->flags & MMAL_BUFFER_HEADER_FLAG_EOS){

		release_semaphore = 1;
		if (pData->debug)
			rpicamera_log_debug("Caught EOS flag., returning control to main thread", NULL);


	}
	else if (pData->capture_complete){
		if(pData->debug)
			rpicamera_log_debug("Received a buffer (flags: %d), but we've already completed the capture.  Doing nothing.", buffer->flags);

	}
	else if (buffer->flags == 0){
		if (pData->debug)
			rpicamera_log_debug("Caught a flush buffer, doing nothing with it.", NULL);
	}

   	else {
      	if (pData->debug){
      		rpicamera_log_debug("Buffer flags:  %0X", buffer->flags);
      		rpicamera_log_debug("Buffer size:  %d, length: %d, offset: %d", bytes_to_write, buffer->length, buffer->offset);
      	}

     	if (pData->bytes_written + bytes_to_write > pData->image_size){
     		rpicamera_log_warning("Possible overflow condition.", NULL);
     		rpicamera_log_warning("  Image size:  %d, bytes already written:  %d", pData->image_size, pData->bytes_written);
     		rpicamera_log_warning("  buffer size:  %d", buffer->length - buffer->offset);
     		bytes_to_write = pData->image_size - pData->bytes_written;
     		// complete = 1;
     	}

      	if (bytes_to_write > 0){
      		if (pData->debug)
      			rpicamera_log_debug("Locking buffer.", NULL);

	       	mmal_buffer_header_mem_lock(buffer);
	       	
	       	if (pData->debug)
				rpicamera_log_debug("writing %d bytes of data to %0x+%d", bytes_to_write, pData->image_data, pData->bytes_written);	       		

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
        	pData->current_frame++;
        	pData->bytes_written = 0;

        	if(pData->debug)
        		rpicamera_log_debug("Current frame incrimented to %d", pData->current_frame);

        	if(pData->current_frame >= pData->num_frames){
        		pData->capture_complete = 1;
        		release_semaphore = 1;
        	}
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

   	// if (complete && (pData->current_frame <= pData->num_frames)) {
   	if (release_semaphore) {
   		if (pData->debug)
   			rpicamera_log_debug("POSTing semaphore", NULL);

      	vcos_semaphore_post(pData->complete_semaphore);
   	}
}

int check_resize_image_buffer(RpiCamera *RpiCamera, MMAL_PORT_T *port){

	MMAL_ES_FORMAT_T *format = port->format;
	PyArrayObject *image_array = (PyArrayObject *)RpiCamera->image;
	PyObject *new_array=NULL;


	int32_t format_size = format->es->video.height * format->es->video.width;
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


PyObject *RpiCamera_integrated_preview(RpiCamera *self, PyObject *arg, PyObject *kwd){

	static char *kwlist[] = {"frames", "shift", NULL};

	// MMAL_ES_FORMAT_T *format = self->output_port->format;
	MMAL_BUFFER_HEADER_T *pool_buffer;
	MMAL_PORT_T *output_port = self->camera->output[MMAL_CAMERA_PREVIEW_PORT];
	INTEGRATED_PREVIEW_USERDATA_T integrated_preview_callback_data;
   	
   	// VCOS_STATUS_T vcos_status = VCOS_SUCCESS;
	uint32_t status = 0;
	uint32_t num_frames = 1;
	uint8_t  shift = 0;
	//Indexes
	uint32_t k = 0;


	//Switch output port to preview port
	PyObject *py_result = NULL;
	if(self->output_port != MMAL_CAMERA_PREVIEW_PORT){
	 	py_result = RpiCamera_switch_output(self, Py_BuildValue("(B)", MMAL_CAMERA_PREVIEW_PORT));

		if(py_result == NULL)
			return NULL;
	}



	if (!PyArg_ParseTupleAndKeywords(arg, kwd, "|IB", kwlist, &num_frames, &shift))
		return NULL;

	if (shift > 7){
		PyErr_Format(PyExc_ValueError, "shift value must be less than 8, provided %d", shift);
		return NULL;
	}

	if (check_resize_image_buffer(self, output_port))
		//There was a problem checking/resizing the image buffer.
		//Errors have already been raised, so just exit.
		return NULL;

	//Clear image object
	PyArray_FILLWBYTE((PyArrayObject *)self->image, 0);


	//Setup callback data structure
	integrated_preview_callback_data.buffer_pool = self->pool;
	integrated_preview_callback_data.bytes_written = 0;
	integrated_preview_callback_data.shift = shift;

	integrated_preview_callback_data.image_size = (uint32_t) PyArray_Size(self->image);
	integrated_preview_callback_data.image_data = (uint8_t *) PyArray_GETPTR1((PyArrayObject *)self->image, 0);
	integrated_preview_callback_data.current_frame = 0;
	integrated_preview_callback_data.num_frames = num_frames;
	integrated_preview_callback_data.capture_complete = 0;
	integrated_preview_callback_data.debug = (uint8_t)self->debug_flag;
	integrated_preview_callback_data.complete_semaphore = self->complete_semaphore;

	// vcos_status = vcos_semaphore_create(&integrated_preview_callback_data.complete_semaphore, "RpiCamera-sem", 0);
	// vcos_assert(vcos_status == VCOS_SUCCESS);

	output_port->userdata = (struct MMAL_PORT_USERDATA_T *)&integrated_preview_callback_data;

	if(self->debug_flag)
		rpicamera_log_debug("Flushing output port", NULL);

	mmal_port_flush(output_port);

	status = mmal_port_enable(output_port, integrated_preview_callback);

	if (status != MMAL_SUCCESS){
		PyErr_SetString(PyExc_RuntimeError, "Could not enable preview output port");
		return NULL;
	}

	//Send all buffers to the output port
	if(self->debug_flag)
		rpicamera_log_debug("Sending %d buffers...", mmal_queue_length(self->pool->queue));

    for (k=0;k<mmal_queue_length(self->pool->queue); k++){
 		pool_buffer = mmal_queue_get(self->pool->queue);
 		if (!pool_buffer)
            rpicamera_log_warning("Unable to get a required buffer %d from pool queue", k);

        if (mmal_port_send_buffer(output_port, pool_buffer)!= MMAL_SUCCESS)
            rpicamera_log_warning("Unable to send a buffer to camera output port (%d)", k);
                 
 	}



 	rpicamera_log_info("Starting capture.", NULL);
 	status = mmal_port_parameter_set_boolean(self->camera->output[MMAL_CAMERA_CAPTURE_PORT], MMAL_PARAMETER_CAPTURE, MMAL_TRUE);
	if(status != MMAL_SUCCESS){
		rpicamera_log_error("%s: Failed to start capture", __func__);
	}

	else{
		// Wait for capture to complete
		// For some reason using vcos_semaphore_wait_timeout sometimes returns immediately with bad parameter error
		// even though it appears to be all correct, so reverting to untimed one until figure out why its erratic
		if(self->debug_flag)
			rpicamera_log_debug("WAITing on sempahore", NULL);
		vcos_semaphore_wait(integrated_preview_callback_data.complete_semaphore);

	}
	status = mmal_port_parameter_set_boolean(self->camera->output[MMAL_CAMERA_CAPTURE_PORT], MMAL_PARAMETER_CAPTURE, MMAL_FALSE);
	integrated_preview_callback_data.capture_complete = 1;
	rpicamera_log_info("Capture finished, got %d frames.", integrated_preview_callback_data.current_frame);

		
	//Delete the semiphore
	// vcos_semaphore_delete(&callback_data.complete_semaphore);


	//Disable the output port
	if(output_port && output_port->is_enabled){
		mmal_port_disable(output_port);
		output_port->userdata = (struct MMAL_PORT_USERDATA_T *) NULL;

	}

	// mmal_port_flush(self->output_port);

	Py_RETURN_NONE;
}


PyObject *RpiCamera_capture_still(RpiCamera *self, PyObject *args, PyObject *kwds){

	// MMAL_ES_FORMAT_T *format = self->output_port->format;
	MMAL_BUFFER_HEADER_T *pool_buffer;
	MMAL_PORT_T *output_port;
	INTEGRATED_PREVIEW_USERDATA_T integrated_preview_callback_data;
   	
   	uint8_t channel = 2; //Default to still port output.
	uint32_t status = 0;
	//Indexes
	uint32_t k = 0;

	static char *kwlist[] = {"channel", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|B", kwlist, &channel))
		return NULL;


	//Switch output port to preview port
	PyObject *py_result = NULL;
	if(self->output_port != channel){
		rpicamera_log_debug("Attempting to switch to port %d.", channel);
	 	py_result = RpiCamera_switch_output(self, Py_BuildValue("(B)", channel));

		if(py_result == NULL)
			return NULL;
	}

	output_port = self->camera->output[self->output_port];
	if (check_resize_image_buffer(self, output_port))
		//There was a problem checking/resizing the image buffer.
		//Errors have already been raised, so just exit.
		return NULL;

	//Clear image object
	PyArray_FILLWBYTE((PyArrayObject *)self->image, 0);


	//Setup callback data structure
	integrated_preview_callback_data.buffer_pool = self->pool;
	integrated_preview_callback_data.bytes_written = 0;
	integrated_preview_callback_data.shift = 0;

	integrated_preview_callback_data.image_size = (uint32_t) PyArray_Size(self->image);
	integrated_preview_callback_data.image_data = (uint8_t *) PyArray_GETPTR1((PyArrayObject *)self->image, 0);
	integrated_preview_callback_data.current_frame = 0;
	integrated_preview_callback_data.num_frames = 1;
	integrated_preview_callback_data.capture_complete = 0;
	integrated_preview_callback_data.debug = (uint8_t)self->debug_flag;
	integrated_preview_callback_data.complete_semaphore = self->complete_semaphore;

	// vcos_status = vcos_semaphore_create(&integrated_preview_callback_data.complete_semaphore, "RpiCamera-sem", 0);
	// vcos_assert(vcos_status == VCOS_SUCCESS);

	output_port->userdata = (struct MMAL_PORT_USERDATA_T *)&integrated_preview_callback_data;

	if(self->debug_flag)
		rpicamera_log_debug("Flushing output port", NULL);

	mmal_port_flush(output_port);

	status = mmal_port_enable(output_port, integrated_preview_callback);

	if (status != MMAL_SUCCESS){
		PyErr_SetString(PyExc_RuntimeError, "Could not enable preview output port");
		return NULL;
	}

	//Send all buffers to the output port
	if(self->debug_flag)
		rpicamera_log_debug("Sending %d buffers...", mmal_queue_length(self->pool->queue));

    for (k=0;k<mmal_queue_length(self->pool->queue); k++){
 		pool_buffer = mmal_queue_get(self->pool->queue);
 		if (!pool_buffer)
            rpicamera_log_warning("Unable to get a required buffer %d from pool queue", k);

        if (mmal_port_send_buffer(output_port, pool_buffer)!= MMAL_SUCCESS)
            rpicamera_log_warning("Unable to send a buffer to camera output port (%d)", k);
                 
 	}

 	rpicamera_log_info("Starting still capture.", NULL);
 	status = mmal_port_parameter_set_boolean(self->camera->output[MMAL_CAMERA_CAPTURE_PORT], MMAL_PARAMETER_CAPTURE, MMAL_TRUE);
	if(status != MMAL_SUCCESS){
		rpicamera_log_error("%s: Failed to start capture", __func__);
	}

	else{
		// Wait for capture to complete
		// For some reason using vcos_semaphore_wait_timeout sometimes returns immediately with bad parameter error
		// even though it appears to be all correct, so reverting to untimed one until figure out why its erratic
		if(self->debug_flag)
			rpicamera_log_debug("WAITing on sempahore", NULL);
		vcos_semaphore_wait(integrated_preview_callback_data.complete_semaphore);

	}
	status = mmal_port_parameter_set_boolean(self->camera->output[MMAL_CAMERA_CAPTURE_PORT], MMAL_PARAMETER_CAPTURE, MMAL_FALSE);
	integrated_preview_callback_data.capture_complete = 1;
	rpicamera_log_info("Still capture finished.", NULL);

		
	//Delete the semiphore
	// vcos_semaphore_delete(&callback_data.complete_semaphore);


	//Disable the output port
	if(output_port && output_port->is_enabled){
		mmal_port_disable(output_port);
		output_port->userdata = (struct MMAL_PORT_USERDATA_T *) NULL;

	}

	// mmal_port_flush(self->output_port);

	Py_RETURN_NONE;
}