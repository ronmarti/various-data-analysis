python-libraries

We use at most python 3.9: as of now, we're limited by pytorch 1.9.1 compatibility.

Consider creating `.env` file in the root directory with content of `PYTHONPATH=.`
so that inside-modules runs scope to the root and the absolute imports
work correctly.