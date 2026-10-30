FROM gcc:13-bookworm@sha256:20eec427045c47d0aad1bd28c81d435916c327cd992240aa78350f755d04d04a

ARG IMAGE_EPOCH=20261030a
RUN echo "itch-image-epoch=${IMAGE_EPOCH}"

# gcc:13-bookworm already ships g++, make, and python3. Bootstrap pip from a
# vendored get-pip.py and install pytest from local wheels (no apt/network).
COPY vendor/get-pip.py /tmp/get-pip.py
COPY vendor/*.whl /tmp/wheels/
RUN python3 /tmp/get-pip.py --break-system-packages --no-index --find-links=/tmp/wheels \
    && pip3 install --break-system-packages --no-cache-dir --no-index --find-links=/tmp/wheels \
        pytest==9.1.1 pytest-json-ctrf==0.5.2 \
    && rm -rf /tmp/get-pip.py /tmp/wheels /root/.cache

COPY . /tests/

RUN mkdir -p /app/src /logs/verifier
