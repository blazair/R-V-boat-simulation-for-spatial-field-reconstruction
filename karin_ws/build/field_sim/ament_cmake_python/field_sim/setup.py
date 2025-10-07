from setuptools import find_packages
from setuptools import setup

setup(
    name='field_sim',
    version='0.0.0',
    packages=find_packages(
        include=('field_sim', 'field_sim.*')),
)
