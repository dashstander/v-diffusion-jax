import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="diffusion",
    py_modules=["diffusion"],
    version="0.1",
    description="",
    author="Katherine Crowson, J.D. Pressman",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
    extras_require={'dev': ['pytest']},
)
