#!/usr/bin/env python
import os
import sys

from setuptools import find_packages, setup
from setuptools.command.install import install

from src import VERSION


class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version"""

    description = "verify that the package git tag matches our version"

    def run(self):
        tag = os.getenv("COMMIT_TAG")

        if tag != VERSION:
            info = "Git tag: {0} does not match the version of this app: {1}".format(
                tag, VERSION
            )
            sys.exit(info)


with open("requirements.txt") as f:
    DEPENDENCIES = f.read().splitlines()

setup(
    name="src",
    packages=find_packages(),
    version=VERSION,
    description="Light weight semantic segmentation.",
    author="Daniel Sola",
    license="MIT",
    install_requires=DEPENDENCIES,
    python_requires=">=3.8",
    url="https://github.com/dansola/PGA-Net",
    cmdclass={"verify": VerifyVersionCommand},
)