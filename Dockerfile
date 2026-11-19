# The whole study in one container: Python 3.12 with the pinned libraries, plus the C++ engine.
#   docker build -t order-flow-signal-research .
#   docker run --rm order-flow-signal-research                      # tests
#   docker run --rm -v "$PWD/repro_out:/src/repro_out" order-flow-signal-research \
#       bash -c "CPP=1 scripts/reproduce.sh"                          # regenerate data and every result
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends g++ cmake make libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY obsr obsr
COPY scripts scripts
COPY tests tests
COPY cpp cpp
RUN cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release && cmake --build cpp/build -j"$(nproc)"
ENV PYTHONPATH=/src
CMD ["python", "-m", "pytest", "tests", "-q"]
