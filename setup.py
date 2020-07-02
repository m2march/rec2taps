#!/usr/bin/env python

from setuptools import setup

setup(name='rec2taps',
      version='0.1',
      description='Utility to obtain tap times from a tapping recording',
      author='Martin "March" Miguel',
      author_email='m2.march@gmail.com',
      packages=['m2', 'm2.rec2taps'],
      namespace_packages=['m2'],
      entry_points={
          'console_scripts': ['rec2taps=m2.rec2taps.cli:rec2taps']
      },
      install_requires=[
          'numpy',
          'scipy'
      ],
      )
