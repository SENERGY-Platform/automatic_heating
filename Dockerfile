# The minor stays 3.10: Ray Client needs the Python of the Ray cluster (rayproject/ray:2.55.0 is py310).
ARG BASE=python:3.10.22-slim@sha256:6ff506466f8b1e981719b468cc0023b8ef000b983cdec0eafe119d01f379e31d

FROM ${BASE} AS revision
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY .git .git
RUN git log -1 --pretty=format:"commit=%H%ndate=%cd%n" > /git_commit

FROM ${BASE}
LABEL org.opencontainers.image.source="https://github.com/SENERGY-Platform/automatic_heating"
WORKDIR /usr/src/app
# git only for the pip install of the operator lib from GitHub
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && python3 -m pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y git && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*
COPY --from=revision /git_commit git_commit
COPY main.py .
COPY algo algo
CMD [ "python3", "-u", "main.py"]
