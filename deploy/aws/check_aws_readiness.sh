#!/usr/bin/env bash
set -euo pipefail

echo "== AWS CLI =="
aws --version

echo
echo "== Authentication check =="
if aws sts get-caller-identity >/dev/null; then
  echo "AWS authentication: OK"
else
  echo "AWS authentication failed."
  exit 1
fi

echo
echo "== Configured region =="
aws configure get region || true

echo
echo "== Existing S3 bucket names =="
aws s3api list-buckets \
  --query "Buckets[].Name" \
  --output table

echo
echo "Read-only checks complete. No resources were created."
