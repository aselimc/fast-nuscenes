from setuptools import setup, find_packages

setup(
    name='fast_nuscenes',
    version = '0.1.1',
    packages=find_packages(),
    install_requires=[
        'nuscenes-devkit',
        'kiss-icp',
    ]
)