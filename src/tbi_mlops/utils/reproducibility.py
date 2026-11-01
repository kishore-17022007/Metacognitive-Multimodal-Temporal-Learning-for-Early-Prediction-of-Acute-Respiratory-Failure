import hashlib, json, os, random, subprocess
import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.use_deterministic_algorithms(False)
    except ImportError:
        pass


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dict_hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


def code_version() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        h = hashlib.sha256()
        for directory, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in {".git", ".venv", ".venv-py315", ".pytest_cache", "__pycache__"})
            for name in sorted(files):
                if name.endswith(".py"):
                    path = os.path.join(directory, name)
                    rel = os.path.relpath(path, root).replace(os.sep, "/")
                    h.update(rel.encode()); h.update(b"\0")
                    with open(path, "rb") as source:
                        h.update(source.read()); h.update(b"\0")
        return f"src-{h.hexdigest()[:12]}"
