FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
WORKDIR /breezyvoice

ENV UV_LINK_MODE=copy
ENV PATH="/root/.local/bin/:$PATH"

ADD https://astral.sh/uv/install.sh /uv-installer.sh

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates ffmpeg&& \
    sh /uv-installer.sh && rm /uv-installer.sh && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    uv venv -p 3.10

COPY requirements.txt /breezyvoice/requirements.txt

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements.txt --index-strategy unsafe-best-match

COPY . .

EXPOSE 8080

ENTRYPOINT ["/breezyvoice/.venv/bin/python"]
