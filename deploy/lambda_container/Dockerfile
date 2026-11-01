FROM public.ecr.aws/lambda/python:3.12

ENV PYTHONPATH="/var/task/src"
ENV MODEL_VERSION="v20261024-171511"
ENV MODEL_BUCKET="tbi-arf-mlops-mlops-2026"
ENV AWS_REGION="ap-southeast-2"

COPY deploy/lambda_container/requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY configs/ ${LAMBDA_TASK_ROOT}/configs/
COPY artifacts/models/manifest.json ${LAMBDA_TASK_ROOT}/manifest.json
COPY deploy/lambda_container/handler.py ${LAMBDA_TASK_ROOT}/handler.py

CMD ["handler.lambda_handler"]
