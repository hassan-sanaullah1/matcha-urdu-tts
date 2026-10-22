# CPU-only server for the kaani male+female Urdu voice. See server/app.py for the API.
#
# This does NOT pip-install the model's Python dependencies (torch, pynini/nemo_text_processing,
# ctranslate2, the compiled matcha monotonic_align .so, ...). Those are already working in the
# pinned venv at ~/Desktop/kaani/envs/venv (built and tested against this exact host), and
# pynini/nemo in particular are slow and fragile to rebuild from a bare pip install. Instead this
# image provides just the system libraries that venv's compiled packages need, and the compose
# file bind-mounts the venv (read-only) plus this repo straight in - no copy, no rebuild, and a
# `pip install` on the host venv is picked up on the next container restart with no image rebuild.
#
# Matching base: the venv's packages (torch, pynini, the Matcha monotonic_align .so compiled for
# cpython-312-x86_64-linux-gnu) were built against the host's Ubuntu 24.04 glibc. A different base
# image's glibc could silently mismatch, so this uses the same release.
FROM ubuntu:24.04

# python3.12 here is ONLY so the bind-mounted venv's symlinks (venv/bin/python3.12 -> .../python3,
# which the venv's own pyvenv.cfg points at /usr/bin/python3.12) resolve inside the container -
# the venv's site-packages supply everything else.
# libespeak-ng1: en_g2p.py loads it in-process via ctypes for English/code-switch phonemes.
# espeak-ng: a SEPARATE code path (src/common/text/phones.py, used by pk_gazetteer.py for
# Pakistani-English spans) shells out to the espeak-ng binary directly, not just the shared lib.
# libsndfile1: soundfile's backend for reading/writing wav.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 libespeak-ng1 espeak-ng libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s python3.12 /usr/bin/python3

WORKDIR /app
EXPOSE 8030

ENV PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1

# --host 0.0.0.0: must listen on all interfaces inside the container for Docker's port
# publishing to reach it (binding to 127.0.0.1 here is unreachable from outside the container,
# same bug the bare-metal run had).
CMD ["/venv/bin/python3.12", "server/app.py", "--host", "0.0.0.0", "--port", "8030"]
