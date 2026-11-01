"""Local version manifest. Versions are immutable; production designation needs an explicit promote() call."""
import json, time
from ..utils.config import ROOT
from ..utils.reproducibility import file_sha256

MODELS = ROOT / "artifacts" / "models"; MANIFEST = MODELS / "manifest.json"


def _load():
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"versions": {}, "production": None}


def register(version: str, files: list, meta: dict):
    m = _load()
    if version in m["versions"]: raise FileExistsError(f"version {version} already registered; refusing to overwrite")
    m["versions"][version] = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": "candidate", "sha256": {f: file_sha256(MODELS / version / f) for f in files}, "meta": meta}
    MODELS.mkdir(parents=True, exist_ok=True); MANIFEST.write_text(json.dumps(m, indent=1)); return m


def promote(version: str, reviewer: str):
    m = _load(); assert version in m["versions"], "unknown version"
    if not reviewer: raise ValueError("promotion requires a named reviewer")
    m["production"] = version; m["versions"][version]["status"] = f"production (promoted by {reviewer})"; MANIFEST.write_text(json.dumps(m, indent=1))


def latest(prefer_production=True):
    m = _load()
    if prefer_production and m["production"]: return m["production"]
    return sorted(m["versions"])[-1] if m["versions"] else None


def verify(version: str) -> bool:
    m = _load()["versions"][version]
    return all(file_sha256(MODELS / version / f) == h for f, h in m["sha256"].items())


def manifest(): return _load()
