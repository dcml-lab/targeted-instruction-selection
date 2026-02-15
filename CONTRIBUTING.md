# Contributing
The repo uses [black](https://github.com/psf/black) AND [isort](https://github.com/PyCQA/isort) for code formatting.
Please setup a pre-commit hook to format the code before each commit.
It helps to minimize the diffs and avoid formatting commits.

Run the following to install the hooks using [pre-commit](https://pre-commit.com/).

```bash
pip install pre-commit
pre-commit install
```