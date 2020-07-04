#!/usr/bin/env python
import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(name='m2-rec2taps',
      version='0.9',
      description='Utility to obtain tap times from a tapping recording',
      long_description=README,
      long_description_content_type='text/markdown',
      url='https://github.com/m2march/rec2taps',
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
