# Codesmith sandbox image.
#
# Minimal Python 3.11. No shell utilities beyond what python-slim ships
# with. Runs as `nobody` (uid 65534). Runtime flags in sandbox.py pin
# everything else (no network, read-only root, mem/cpu/pids limits).
#
# Build:
#   docker build -t codesmith-sandbox:latest -f docker/sandbox.Dockerfile docker/
#
# If you need more libs available to agent-generated code, add them here:
#   RUN pip install --no-cache-dir numpy pandas requests ...
# Be aware: more libs = bigger attack surface. Start minimal.

FROM python:3.11-slim

# A few commonly useful stdlib-adjacent packages. Add more sparingly.
# These run at build time as root; the container will still run as nobody.
RUN pip install --no-cache-dir --root-user-action=ignore \
        pytest==8.3.3 \
    && rm -rf /root/.cache/pip

# /work is where the session workspace gets bind-mounted at run time.
RUN mkdir -p /work && chmod 777 /work

WORKDIR /work
USER nobody

# Default command is a placeholder — sandbox.py always passes `command=`
# at run time, overriding this.
CMD ["python", "-c", "print('sandbox image ok')"]
