from importlib.metadata import entry_points
from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

import os

if os.name =='nt':
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

setup(
    name='morphdist',
    version='0.01',
    description='Set of tools and notebooks for distance calculations of morphological data',
    long_description=readme(),
    classifiers=[
        'Development Status :: Number - Alpha',
        'License :: OSI Approved :: MIT License', #GNU technically
        'Programming Language :: Python :: 3.8'
    ],
    keywords='datascience distance',
    url='https://github.com/jzsexton/Morphologic-Distance',
    author='bhalliga',
    author_email='bhalliga@med.umich.edu',
    license='MIT',
    packages=['explorer'],
    entry_points = {
        # 'console_scripts' : []
    },
    install_requries=[
        'sqlite3', 'sklearn', 'scipy'
    ],
    dependency_links=[],
    include_package_data=True,
    zip_safe=False,
    test_suite='nose.collector',
)