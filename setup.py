from setuptools import setup, Extension

RpiCamera = Extension('RpiCamera',
    include_dirs=['./externals/rpi-userland',
                  './externals/rpi-userland/interface/vcos/pthreads',
                  './externals/rpi-userland/interface/mmal'],
    libraries=['mmal_core','mmal_util', 'mmal_vc_client', 'vcos', 'bcm_host'],
    library_dirs=['/opt/vc/lib'],
    sources=['RpiCamera/RpiCamera.c',
              'RpiCamera/camera_settings.c'])


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
