from setuptools import setup
from distutils.core import Command

with open("README.md") as fh:
    long_description = fh.read()
setup(
    name='augmentiX',
    version='0.0.10',
    packages=['augmentix'],
    url='https://github.com/suryan-s/AugmentiX',
    entry_points={
        'console_scripts': [
            'augmentix = augmentix.main:main',
            ]
        },
    license='MIT',
    author='s-suryan',
    author_email='suryannasa@gmail.com',
    description='AugmentiX is a image augmentation library for populating YOLO datasets',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Software Development :: Libraries",
        "Operating System :: OS Independent",
        ],
    install_requires=[open('requirements.txt').read().strip().split('\n')],
    python_requires='>=3.10',
    )