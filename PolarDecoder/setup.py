#!/usr/bin/env python
import os
import re
import sys
import subprocess as sp
from datetime import datetime

import setuptools
from setuptools.command.build_ext import build_ext
from setuptools import setup, find_packages, Extension

HERE = os.path.dirname(__file__)

def read(fname):
    return open(os.path.join(HERE, fname)).read()


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            sp.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                ", ".join(e.name for e in self.extensions)
            )

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        import numpy
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable
                      ]

        cfg = 'Release'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args += ['--', '-j8']

        env = os.environ.copy()
        env['CXXFLAGS'] = (
            '{} -I{} -DVERSION_INFO=\\"{}\\"'
            .format(
                env.get('CXXFLAGS', ''),
                numpy.get_include(),
                self.distribution.get_version()
            )
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        sp.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        sp.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

VERSION = "0.0.1"
setup(
    name="libPolarDecoder",
    version=VERSION,
    author="CaoZhiWei",
    description=("Implement traditional SC series decoder (SC/SCL/FastSC/FastSCL) and LUT SC decoder (SCLUT/SCLLUT/FastSCLUT/FastSCLLUT)"),
    license="WTF",
    packages=find_packages(include=("PolarDecoder*", )),
    ext_modules=[CMakeExtension("PolarDecoder._cpp._libPolarDecoder")],
    install_requires=[
        'numpy',
    ],
    cmdclass={'build_ext': CMakeBuild},
    include_package_data=False,
)
# vim: ts=4 sw=4 sts=4 expandtab
