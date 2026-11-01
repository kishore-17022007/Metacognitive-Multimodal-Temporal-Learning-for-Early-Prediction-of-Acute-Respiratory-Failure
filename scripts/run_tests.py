"""Minimal pytest-compatible runner for environments without pytest. `pytest -q` also works when pytest is installed."""
import sys, pathlib, importlib, traceback, time, unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (ROOT / "tests", ROOT / "src", ROOT): sys.path.insert(0, str(p))
passed = failed = skipped = 0; fails = []
for f in sorted((ROOT / "tests").glob("test_*.py")):
    mod = importlib.import_module(f.stem)
    for name in sorted(n for n in dir(mod) if n.startswith("test_")):
        t0 = time.time()
        try: getattr(mod, name)(); passed += 1; print(f"PASS  {f.stem}::{name} ({time.time()-t0:.1f}s)")
        except unittest.SkipTest as e: skipped += 1; print(f"SKIP  {f.stem}::{name} - {e}")
        except Exception: failed += 1; fails.append(f"{f.stem}::{name}"); print(f"FAIL  {f.stem}::{name}"); traceback.print_exc()
print(f"\n{passed} passed, {failed} failed, {skipped} skipped"); sys.exit(1 if failed else 0)
