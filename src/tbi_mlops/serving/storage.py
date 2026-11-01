"""Artifact storage interface. Local is implemented/tested; S3 is a documented stub for the later reviewed AWS phase."""
import os, pathlib, shutil


class ArtifactStorage:
    def get(self, key: str, dest: str): raise NotImplementedError
    def put(self, src: str, key: str): raise NotImplementedError


class LocalStorage(ArtifactStorage):
    def __init__(self, root): self.root = pathlib.Path(root)
    def get(self, key, dest): shutil.copy(self.root / key, dest)
    def put(self, src, key): (self.root / key).parent.mkdir(parents=True, exist_ok=True); shutil.copy(src, self.root / key)


class S3Storage(ArtifactStorage):          # NOT used or tested; no cloud resources are created in this phase
    def __init__(self, bucket, prefix=""):
        import boto3                          # lazy import; credentials come from the standard AWS provider chain
        self.s3, self.bucket, self.prefix = boto3.client("s3"), bucket, prefix
    def get(self, key, dest): self.s3.download_file(self.bucket, self.prefix + key, dest)
    def put(self, src, key): self.s3.upload_file(src, self.bucket, self.prefix + key)


def get_storage(root):
    return S3Storage(os.environ["TBI_S3_BUCKET"]) if os.environ.get("TBI_ARTIFACT_BACKEND") == "s3" else LocalStorage(root)
