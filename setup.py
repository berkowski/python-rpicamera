from setuptools import setup, Extension

RpiCamera = Extension('RpiCamera',
    include_dirs=['./externals/rpi-userland',
                  './externals/rpi-userland/interface/vcos',
                  './externals/rpi-userland/interface/vcos/pthreads',
                  './externals/rpi-userland/interface/mmal',
                  './externals/rpi-userland/interface/vmcs_host/linux'],
    libraries=['mmal_core','mmal_util', 'mmal_vc_client', 'vcos', 'bcm_host'],
    library_dirs=['/opt/vc/lib'],
    sources=['RpiCamera/RpiCamera.c',
              'RpiCamera/RpiCamera_settings.c',
              'RpiCamera/RpiCamera_capture.c',
              'RpiCamera/RpiCamera_logging.c'])


setup (name = 'RpiCamera',
    version = '1.0',
    description = 'Python module for taking still images with the Raspberry Pi Camera Module.',
    author = 'Zac Berkowitz',
    author_email = 'zac.berkowitz@gmail.com',
    url = 'http://docs.python.org/extending/building',
    ext_modules = [RpiCamera])       
