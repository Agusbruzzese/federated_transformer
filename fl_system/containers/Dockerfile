# fl-base (report 28): Ubuntu 24.04 + the same python3.12 package version as the validated host environment.
# The Python environments (.venv-fl, .venv) and the code are mounted read-only at run time; data folders are mounted per container.
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends python3.12=3.12.3-1ubuntu0.17 libpython3.12-stdlib=3.12.3-1ubuntu0.17 \
        python3.12-minimal=3.12.3-1ubuntu0.17 libpython3.12-minimal=3.12.3-1ubuntu0.17 libgomp1 iproute2 \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 HOME=/tmp PYTHONUNBUFFERED=1
