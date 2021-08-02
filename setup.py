import os
import pathlib
from setuptools import setup

_HERE = pathlib.Path(__file__).parent

setup(
    name="potrace_py",
    version="0.0.1",
    description="Port of potrace written in python",
    long_description=open(os.path.join(_HERE, "README.md"), "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mahdilamb/potrace_py",
    author="Mahdi Lamb",
    author_email="mahdilamb@gmail.com",
    license_files=("LICENSE.txt",),
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",

    ],
    packages=["potrace"],
    include_package_data=True,
    install_requires=["numpy"],

)
