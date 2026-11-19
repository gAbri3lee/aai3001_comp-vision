FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
COPY cloud/requirements.txt /app/cloud/requirements.txt
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch==2.9.0 torchvision==0.24.0 --index-url ${TORCH_INDEX_URL} && \
    pip install --no-cache-dir -r /app/cloud/requirements.txt
COPY src /app/src
COPY visionaid_config.json /app/visionaid_config.json
ENV PYTHONPATH=/app/src VISIONAID_DEVICE=cpu
EXPOSE 9000
# Mount the supplied checkpoint at /app/models/cloud/rfdetr-nano_v4_four_class.pth.
# One worker loads the model once; the app rejects concurrent inference.
CMD ["gunicorn", "--bind", "0.0.0.0:9000", "--workers", "1", "--threads", "2", "--timeout", "120", "cloud_service:create_app()"]
