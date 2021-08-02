import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()
LICENCE = (HERE / "LICENSE").read_text()

setup(
    name="potrace_py",
    version="0.0.1",
    description=" Port of potrace written in python",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/mahdilamb/potrace_py",
    author="Mahdi Lamb",
    author_email="mahdilamb@gmail.com",
    license=LICENCE,
    classifiers=[
        "License :: OSI Approved :: GPL-3.0-or-later",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",

    ],
    packages=["potrace"],
    include_package_data=True,
    install_requires=["numpy"],

)
