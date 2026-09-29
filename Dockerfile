FROM nvcr.io/nvidia/pytorch:25.01-py3
WORKDIR /workspace/ttt
COPY pyproject.toml requirements.txt requirements-agents.txt ./
RUN python -m pip install --no-cache-dir -r requirements-agents.txt
COPY . .
RUN python -m pip install --no-cache-dir -e .
ENV PYTHONPATH=/workspace/ttt
CMD ["python", "scripts/selfcheck.py"]
