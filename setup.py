from setuptools import setup, find_packages

"""
The first time you need to run:
 pip install -e .
"""

setup(
    name='myProject',
    description='',
    packages=find_packages(include=["MyProject", "MyProject.*"]),
    version="0.1")
