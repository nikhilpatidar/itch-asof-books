FROM gcc:13-bookworm@sha256:20eec427045c47d0aad1bd28c81d435916c327cd992240aa78350f755d04d04a

ARG IMAGE_EPOCH=20261030a
RUN echo "itch-image-epoch=${IMAGE_EPOCH}"

WORKDIR /app

COPY Makefile /app/Makefile
COPY src /app/src

RUN mkdir -p /app/scratch && make -C /app
