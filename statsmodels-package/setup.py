from setuptools import setup

setup (
    name="statsmodels_package",
    version = "0.1",
    author="Martin Ron",
    author_email="ronmarti@fel.cvut.cz",
    description = "Collection of statistical models for productivity use.",
    packages=['statsmodels_collection'],
    python_requires=">=3.8",
    )