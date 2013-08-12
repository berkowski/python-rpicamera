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
    description = 'This is a demo package',
    author = 'Martin v. Loewis',
    author_email = 'martin@v.loewis.de',
    url = 'http://docs.python.org/extending/building',
    long_description = '''
This is really just a demo package.
''',
    ext_modules = [RpiCamera])       
