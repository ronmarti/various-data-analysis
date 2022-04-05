## Python Package

To install locally, run within the root of the package `statsmodels_collection` following:
```
...statsmodels_collection> pip install -e .
```
That installs the package into the current python interpretter.
The important bit is the `setup.py` file, that is used to configure packaging. To deploy to pypi, you will need it.

Sources: https://medium.com/pythoneers/how-to-make-your-own-python-package-61abf012ac96