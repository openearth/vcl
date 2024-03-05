#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "Click>=7.0",
    "pyzmq",
    "xarray",
    "rioxarray",
    "scipy",
    "numpy",
    "netCDF4",
    "cmocean",
    "scipy",
    "opencv-python",
    "geopandas",
    "rasterio",
    "matplotlib",
    "shapely",
    "pandas",
    "mido",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Fedor Baart",
    author_email="fedor.baart@deltares.nl",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Virtual Climate Lab: an interactive table to discuss the future of your environment.",
    entry_points={
        "console_scripts": [
            "vcl=vcl.cli:main",
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="vcl",
    name="vcl",
    packages=find_packages(include=["vcl", "vcl.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/openearth/vcl",
    version="0.1.0",
    zip_safe=False,
)
