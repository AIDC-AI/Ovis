from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'requirements.txt'), 'r', encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name='ovis',
    version='2.0.0',
    packages=find_packages(where='.', exclude=('tests', 'docs')),
    install_requires=requirements
)

