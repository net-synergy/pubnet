# syntax=docker/dockerfile:1
FROM python:3.12

RUN pip install --upgrade pip && \
    pip install pipx && \
    pipx install poetry

ENV PATH="$PATH:/root/.local/bin"
ENV POETRY_VIRTUALENVS_IN_PROJECT=true

CMD ["/bin/bash", "-c", "poetry install --with=test --all-extras >/dev/null && poetry run pytest ."]
