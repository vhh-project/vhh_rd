import logging
import os
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

# read requirements
install_requires=[]
with open("requirements.txt", "r") as f:
    reqs = f.readlines()
    for i in range(len(reqs)):
        req = reqs[i]
        if i < len(reqs)-1:
            req = req[:-1]
        if req[0] is not '#':
            install_requires.append(req)

# install vhh_rd package
setup(
     name='vhh_rd',
     version='1.0.0',
     author="Daniel Helm",
     author_email="daniel.helm@tuwien.ac.at",
     description="ORelation Detection Package",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/dahe-cvl/vhh_rd",
     packages=find_packages(),
     install_requires=install_requires
)
