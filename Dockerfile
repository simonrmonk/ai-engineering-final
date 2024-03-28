FROM python:3.11

# RUN useradd -m -u 1000 user
RUN useradd -ms /bin/bash appuser

RUN apt-get update -y \
    && apt-get install -y make curl \
    && curl -sSL https://install.python-poetry.org | POETRY_HOME=/home/appuser/.local python3 - \
    # && curl -sSL https://install.python-poetry.org | python3 - \
    # && curl -sSL https://install.python-poetry.org | python3 -  \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/home/appuser/.local/bin:$PATH"

RUN poetry config virtualenvs.create false

RUN mkdir /home/appuser/demoday

COPY poetry.lock /home/appuser/demoday
COPY pyproject.toml /home/appuser/demoday

ENV HOME=/home/appuser/demoday
WORKDIR $HOME
RUN chown -R appuser:appuser $HOME
RUN poetry install
WORKDIR $HOME/demoday

COPY . /home/appuser/demoday
COPY FPL_csvs /home/appuser
RUN poetry install --sync

RUN chown -R appuser:appuser $HOME

USER appuser
WORKDIR $HOME

CMD ["poetry", "run", "chainlit", "run", "app.py", "--port", "7860"]
