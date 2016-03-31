#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import sys
import subprocess


MAJOR = 0
MINOR = 1
MICRO = 1
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


# Return the git revision as a string
def svn_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['svn',  'info', '--show-item', 'revision'])
        SVN_REVISION = out.strip().decode('ascii')
    except OSError:
        SVN_REVISION = "Unknown"

    return SVN_REVISION


def get_version_info():
    # Adding the git rev number needs to be done inside
    # write_version_py(), otherwise the import of pydespiking.version messes
    # up the build under Python 3.
    FULLVERSION = VERSION
    if os.path.exists('pydespiking/version.py'):
        # must be a source distribution, use existing version file
        # load it as a separate module to not load pywt/__init__.py
        import imp
        version = imp.load_source('pydespiking.version', 'pydespiking/version.py')
        SVN_REVISION = version.svn_revision
    else:
        SVN_REVISION = svn_version()

    if not ISRELEASED:
        FULLVERSION += '.dev0+' + SVN_REVISION[:7]

    return FULLVERSION, SVN_REVISION


def write_version_py(filename='pydespiking/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM pydespiking SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
svn_revision = '%(svn_revision)s'
release = %(isrelease)s

if not release:
    version = full_version
"""
    FULLVERSION, SVN_REVISION = get_version_info()

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'svn_revision': SVN_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()


# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')


if sys.platform == "darwin":
    # Don't create resource files on OS X tar.
    os.environ["COPY_EXTENDED_ATTRIBUTES_DISABLE"] = "true"
    os.environ["COPYFILE_DISABLE"] = "true"


setup_args = {}



def generate_cython():
    cwd = os.path.abspath(os.path.dirname(__file__))
    print("Cythonizing sources")
    p = subprocess.call([sys.executable,
                          os.path.join(cwd, 'util', 'cythonize.py'),
                          'pydespiking'],
                         cwd=cwd)
    if p != 0:
        raise RuntimeError("Running cythonize failed!")


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('pydespiking')

    config.get_version('pydespiking/version.py')
    return config


def setup_package():

    # Rewrite the version file everytime
    write_version_py()

    metadata = dict(
        name="pydespiking",
        maintainer="Holger Nahrstaedt",
        maintainer_email=" ",
        url="",
        download_url="",
        license="BSD",
        description="library for removing spike noise",
        long_description="""\
       pydespiking is a small python library for removing spike noise. 

        """,
        keywords=["pydespiking"],
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Operating System :: OS Independent",
            "Programming Language :: C",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 2.6",
            "Programming Language :: Python :: 2.7",
            "Programming Language :: Python :: 3.3",
            "Programming Language :: Python :: 3.4",
            "Programming Language :: Python :: 3.5",
            "Topic :: Software Development :: Libraries :: Python Modules"
        ],
        platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
        cmdclass={},
        **setup_args
    )
    if len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
            sys.argv[1] in ('--help-commands', 'egg_info', '--version',
                            'clean')):
        # For these actions, NumPy is not required.
        #
        # They are required to succeed without Numpy for example when
        # pip is used to install PyWavelets when Numpy is not yet present in
        # the system.
        try:
            from setuptools import setup
        except ImportError:
            from distutils.core import setup

        FULLVERSION, SVN_REVISION = get_version_info()
        metadata['version'] = FULLVERSION
    else:
        if (len(sys.argv) >= 2 and sys.argv[1] == 'bdist_wheel') or (
                    'develop' in sys.argv):
            # bdist_wheel needs setuptools
            import setuptools

        from numpy.distutils.core import setup

        cwd = os.path.abspath(os.path.dirname(__file__))
        #if not os.path.exists(os.path.join(cwd, 'PKG-INFO')):
            # Generate Cython sources, unless building from source release
            #expand_src_templates()
            #generate_cython()

    metadata['configuration'] = configuration

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
