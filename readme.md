RpiCamera
=========

RpiCamera is a Python extension for capturing still images with 
the Raspberry Pi camera module.  

Quick and dirty features:
    
    *  Single frame capture up to full sensor resolution in YUV420p or BGR24 formats
    *  Multiple frame integration
    *  Access to camera settings such as brightness, saturation, etc.
    *  Logging using Python's logging module.

Introduction
------------

This module came about while investigating the feasability for using the 
Raspberry Pi camera module in a camera instrument.  We were interested being
able to quickly access propperly exposed raw frame data, with an emphasis
on intensity data (less so actual colored images).  As such this module provides
ways of pulling raw data from the camera sensor in YUV420p or BGR24 formats, what
you do with it after is up to you!

Requirements
------------

*  Python 3.x.  (Should work w/ 2.7, but has not been tested)
*  Raspberry Pi userland headers and libraries. 

Recommendations
---------------

There will be bugs; this is my first C extension for Python and I will be
losing access to the RPi + Camera module shortly.  It is *highly* recommended
that you install this module inside a virtualenv.


Building
--------

Clone the repository and grab the Raspberry Pi userland sources::

    git clone https://github.com/berkowski/python-rpicamera.git
    git submodule init
    git submodule update

The submodule will take care of putting the necessary C headers where they're
expected.  Libraries are expected to reside in /opt/vc/lib.  If you've installed
the userland libraries someplace else, change the `library_dirs` list in `setup.py`
to point to the correct location.

Build the module using the provided Python setup.py script::

    python setup.py build
    python setup.py install

Again; highly recommended you do this in a virtualenv.


Usage
-----

Basic usage follows :

    #.  Create a camera instance
    #.  Set an output format
    #.  Adjust some camera parameters
    #.  Capture some data

Camera instances are easily created with::

    >>> c = RpiCamera.Camera()

Though only one camera instance can be active at a time.

Use the `set_output_format` method to set an output format, see the docstring for details.
But for an example, here we set the output format to YUV_420 at the maximum size
of 2624x1956 (which includes the blanking rows/columns)::

    >>> c.set_output_format(RpiCamera.CAMERA_STILL, width=2624, height=1956, width_offset=0,
            height_offset=0, encoding=RpiCamera.YUV_420I)

Camera settings are exposed as various attributes such as `brightness`, or `saturation`
on the camera object.  Currently the following attributes can be adjusted:

    * saturation
    * sharpness
    * contrast
    * brightness
    * iso
    * metering_mode
    * exposure_compensation
    * video_stabilisation
    * exposure_mode
    * awb_mode
    * image_fx

Right now only saturation, sharpness, contrast, and iso are straight-forward to adjust.
The rest require enumeration paramters I haven't gotten around to exposing to the user.

There are two methods of aquiring images: capturing a single image, or integrating
multiple images.  Single-frame imaging can use either the RpiCamera.CAMERA_STILL port OR
the RpiCamera.CAMERA_PREVIEW port.  Multi-frame integration can ONLY use the 
RpiCamera.CAMERA_PREVIEW port.  Here we capture a single frame using the format we
just defined on CAMERA_STILL

    >>> c.capture_still_image(RpiCamera.CAMERA_STILL)

Data from the last capture is always stored on the 1D numpy buffer attribute `image`, 
so our single frame data is now at
    
    >>> c.image
        array([156, 170, 166, ..., 108, 118, 118], dtype=uint8)

Check the examples section next for ways to look at these images.

Examples
--------

Capture a single 1024x768 frame in BGR24 format from the CAMERA_STILL output port, 
display the result using matplotlib::

    from matplotlib import pyplot as plt
    import RpiCamera

    c = RpiCamera.Camera()
    c.set_output_format(RpiCamera.CAMERA_STILL, width=1024, height=768, width_offset=0,
        height_offset=0, crop_width=1024, crop_height=768, encoding=RpiCamera.BGR24)

    c.capture_still_image(RpiCamera.CAMERA_STILL)

    #BGR24 1D to 768x1024x3 image:
    #reshape provides the 3x1024x768 matrix, the transpose gives 768x1024x3
    #order='F' is important here!
    img = c.image.reshape(3, 1024, 768, order='F').T

    #Need to reverse the color order, imshow expects RGB, not BGR
    plt.imshow(img[:, :, -1::-1])
    plt.show()  

Integrate 20 frames at 640x480 @ 60 FPS using YUV420::

    from matplotlib import pyplot as plt
    import RpiCamera

    #We'll show some logging too
    import logging
    logging.basicConfig(level=logging.DEBUG, 
        format='%(asctime)-15s %(name)s|%(levelname) 8s:: %(message)s')

    c = RpiCamera.Camera()

    #Frame integration use the PREVIEW port on the camera, so here we adjust that
    #port's format
    c.set_output_format(RpiCamera.CAMERA_PREVIEW, width=640, height=480, width_offset=0,
        height_offset=0, crop_width=640, crop_height=480, frame_rate_num=60,
        frame_rate_den=1, encoding=RpiCamera.YUV_420I)

    c.integrate_preview_frames(frames=20, shift=2)

    #YUV_420 1D to 480x640 image:
    #Only interested in the intensity here, the 'Y' values,
    #which make up the first 480*640 'pixels'.
    #order='F' is important here!
    img = c.image[:480*640].reshape(480, 640, order='C')

    #Need to reverse the color order, imshow expects RGB, not BGR
    plt.imshow(img, cmap='gray')
    plt.show()

Logging
-------
The RpiCamera module uses the python logging module with a logger name 'RpiCamera'. See
http://docs.python.org/3.3/library/logging.html for information using the logging module.

A Note on Formats
-----------------

You can use the set_output_fomat method to create arbitrary-sized output images, but some
care is required when building the two-dimension array from the resulting image buffer.

For example, say you wanted an 80x80 image, so you try::

    >>> c.set_output_format(RpiCamera.CAMERA_STILL, width=80, height=80)
    >>> c.get_output_format(RpiCamera.CAMERA_STILL)
        {'crop_height': 80, 'crop_width': 80, 'encoding': 808596553,
            'frame_rate_den': 1, 'frame_rate_num': 15, 'height': 80,
            'height_offset': 0, 'width': 96, 'width_offset': 0}

What happened?  You wanted a width of 80, but a width of 96 was set.  Now, when you capture
an image, the image dimensions will be 96x80 instead of 80x80!  HOWEVER, only the upper
80x80 rectangle will have data! (note the crop_width=80 !)

So while you can get arbitrary sized *images*, you cannot get arbitrary sized *buffers*.
When playing around with image formats it's always useful to check the format after
setting to make sure you know what the actual data dimensions are.
