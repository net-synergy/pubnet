# PubNet publication networks

Provides data types for managing publication networks as a set of
graphs.

`PubNet` provides functions for downloading, storing, manipulating, and
saving publication networks. Networks can come from common online
sources like pubmed and crossref.

## Installation

``` bash
pip install --user pubnet
```

## More help

See [Documentation](https://net-synergy.gitlab.io/pubnet)

## Running tests

Installing with `poetry install --with=dev,tests --all-extras` will install all dependencies needed for running the tests. Then `poetry run pytest .` will run the test suite.

Optionally, if poetry is not installed, there is a Dockerfile provided. Build the image with `docker build --tag poetry:3.12 .` then the image can be run with `docker run --rm -it -v $PWD:/home/pubnet -w /home/pubnet poetry:3.12`. The initial run will download dependencies to `.venv` inside the directory. After the first run, the dependencies will be reused. It will then run the tests on the current work tree.

This container can also be used to run arbitrary commands without the need to install poetry directly by passing a command to the end of the docker command:

``` bash
docker run --rm -it -v $PWD:/home/pubnet -w /home/pubnet /bin/bash
```

In this case, if `poetry install` hasn't already been run, this will need to be run with the above install command. Then, inside the container, `poetry run ipython` can be run to start a python repl with pubnet installed.
