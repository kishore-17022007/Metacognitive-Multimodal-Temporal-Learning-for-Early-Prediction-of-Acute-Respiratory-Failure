# AWS Candidate Training

AWS mode is opt-in and trains only the scikit-learn reliability stack. It downloads the two required synthetic CSVs, writes model files to a temporary local directory, uploads an immutable candidate bundle, and conditionally merges a candidate record into the S3 manifest. It never promotes a model or writes to the local artifact registry.

## Inputs and outputs

By default, the job reads:

- `s3://$TBI_S3_BUCKET/datasets/synthetic/raw_observations.csv`
- `s3://$TBI_S3_BUCKET/datasets/synthetic/patients.csv`

It publishes the eight inference artifacts under `s3://$TBI_S3_BUCKET/models/<model-version>/`, updates `models/manifest.json` using S3 conditional writes, and stores the JSON run report under `models/training-reports/<training-run-id>.json`. Missing input objects fail the run; AWS mode never falls back to generated data.

The report is created before training and conditionally updated through `RUNNING`, `CANDIDATE_ARTIFACTS_UPLOADED`, `CANDIDATE_REGISTERED`, and finally `SUCCEEDED_CANDIDATE` with `training_report_status: COMPLETED`. Registration does not happen until all eight artifact uploads and the uploaded-artifact report update succeed. If an artifact upload fails after some objects were written, the failure report identifies the uploaded filenames, failed artifact, and orphaned candidate prefix. No S3 objects are automatically deleted. If the final report update fails, the candidate may be registered, but the last durable report state remains not completed and the command raises an error.

## Environment and command

Required:

- `TBI_S3_BUCKET`: bucket containing the input data and model outputs.

Optional prefixes and region:

- `TBI_S3_INPUT_PREFIX`: input prefix; default `datasets/synthetic`.
- `TBI_S3_MODEL_PREFIX`: output prefix; default `models`.
- `AWS_REGION` or `AWS_DEFAULT_REGION`: region; otherwise use the SDK's configured region.
- `TBI_PROFILE`: local YAML profile; default `dev`.

Use the standard AWS credential provider chain (for example, an execution role or configured SSO profile). Do not put credentials in the command, source, or repository.

```sh
TBI_S3_BUCKET=your-isolated-test-bucket \
TBI_S3_INPUT_PREFIX=datasets/synthetic \
TBI_S3_MODEL_PREFIX=models \
AWS_REGION=ap-southeast-2 \
python scripts/train_advanced.py --aws --skip-torch
```

Run the read-only preflight before training to check the current identity, region, input CSV headers, and isolated output-prefix listing permission. It does not create, upload, update, or delete S3 objects. Set `TBI_EXPECTED_AWS_ACCOUNT` if the preflight should reject a different account.

```sh
TBI_S3_BUCKET=tbi-arf-mlops-mlops-2026 \
TBI_S3_INPUT_PREFIX=datasets/synthetic \
TBI_S3_MODEL_PREFIX=models-aws-training-test \
AWS_REGION=ap-southeast-2 \
TBI_EXPECTED_AWS_ACCOUNT=283912199308 \
.venv/bin/python scripts/aws_training_preflight.py
```

`--skip-torch` is required with `--aws`; `--version` is disallowed because AWS mode generates a timestamp-plus-random model version for every run. Omitting `--aws` preserves the existing local behavior.

For the first real AWS candidate-training test in this project, use this isolated output prefix so the existing `models/` registry is not touched:

```sh
TBI_S3_BUCKET=tbi-arf-mlops-mlops-2026 \
TBI_S3_INPUT_PREFIX=datasets/synthetic \
TBI_S3_MODEL_PREFIX=models-aws-training-test \
AWS_REGION=ap-southeast-2 \
python scripts/train_advanced.py --aws --skip-torch
```

## Minimum S3 permissions

This exact policy is for the isolated test configuration below. The training code requires `GetObject` on both inputs and the test manifest, plus `PutObject` only below the isolated output prefix. The preflight additionally uses `ListBucket` scoped to that output prefix to verify that it can list there; `ListBucket` is not used by training itself. Conditional S3 headers require no additional IAM action. No delete permission is granted.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject"],
      "Resource": [
        "arn:aws:s3:::tbi-arf-mlops-mlops-2026/datasets/synthetic/*",
        "arn:aws:s3:::tbi-arf-mlops-mlops-2026/models-aws-training-test/manifest.json"
      ]
    },
    {
      "Effect": "Allow",
      "Action": ["s3:PutObject"],
      "Resource": ["arn:aws:s3:::tbi-arf-mlops-mlops-2026/models-aws-training-test/*"]
    },
    {
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": ["arn:aws:s3:::tbi-arf-mlops-mlops-2026"],
      "Condition": {
        "StringLike": {
          "s3:prefix": ["models-aws-training-test/", "models-aws-training-test/*"]
        }
      }
    }
  ]
}
```

If the bucket enforces SSE-KMS with a customer-managed key, the execution role also needs the corresponding KMS encrypt/data-key permissions for writes and decrypt permissions for reads.

## Safe synthetic-data test

Use a non-production bucket and a fresh test prefix. Upload copies of the repository's synthetic CSVs into that prefix, set `TBI_S3_INPUT_PREFIX` to it, and set `TBI_S3_MODEL_PREFIX` to a separate test output prefix. Run the command above and verify the candidate's eight files, manifest entry (`status: candidate`), and training report. The job does not promote or alter the inference deployment. Do not point this test at the production model prefix if the goal is to keep its manifest unchanged.

The candidate bundle uses the same filenames and formats consumed by `serving/predictor.py`. The running inference API will not select this candidate unless a separate, reviewed deployment/promotion process updates its pinned model version and manifest.