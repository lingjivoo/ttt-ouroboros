ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.01-py3
FROM ${BASE_IMAGE}

WORKDIR /workspace/ttt-feedback
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace/ttt-feedback \
    TTT_OUT=/workspace/ttt-feedback/results

COPY pyproject.toml requirements.txt requirements-agents.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements-agents.txt
COPY . .
RUN python -m pip install --no-cache-dir -e . && \
    python -m compileall -q ttt_pt scripts analysis validation

# Mount checkpoints/data and pass --suite/--smoke as needed.
ENTRYPOINT ["python", "scripts/run_paper_suite.py"]
CMD ["--help"]
