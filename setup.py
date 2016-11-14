import os
import logging
from setuptools import setup
from setuptools import find_packages

setup(name='timeflow',
      version='0.1',
      description='Library for using tensorflow for time series',
      url='https://github.com/abhishekmalali/TimeFlow',
      author='Abhishek Malali, Pavlos Protopapas',
      author_email='anon@anon.com',
      license='MIT',
      include_package_data=True,
      packages=find_packages(),
      install_requires=['numpy', 'scipy', 'nose', 'matplotlib'])
