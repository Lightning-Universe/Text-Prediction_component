#!/usr/bin/env python
import os

from pkg_resources import parse_requirements
from setuptools import find_packages, setup

_PATH_ROOT = os.path.dirname(__file__)

with open(os.path.join(_PATH_ROOT, "README.md")) as fo:
    readme = fo.read()


def _load_requirements(path_dir: str = _PATH_ROOT, file_name: str = "requirements.txt") -> list:
    reqs = parse_requirements(open(os.path.join(path_dir, file_name)).readlines())
    return list(map(str, reqs))


setup(
    name="lai-textpred",
    version="0.0.1",
    description="LAI components for training text prediction",
    author="Lightning-AI",
    author_email="",
    long_description=readme,
    long_description_content_type="text/markdown",
    python_requires =">=3.8",
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url="https://github.com/PyTorchLightning/lightning-component-template",
    install_requires=_load_requirements(),
    packages=find_packages(exclude=["tests", "tests.*"]),
    zip_safe = False,
    include_package_data = True,
    classifiers=[
        "Environment :: Console",
        "Natural Language :: English",
        # How mature is this project? Common values are
        #   3 - Alpha, 4 - Beta, 5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        # Pick your license as you wish
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
