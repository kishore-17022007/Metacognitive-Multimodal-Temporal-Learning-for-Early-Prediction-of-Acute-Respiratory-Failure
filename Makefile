PY ?= python
.PHONY: venv install data prepare baseline advanced simulate evaluate test verify mlflow api dashboard all
venv:       ; $(PY) -m venv .venv && echo "activate: source .venv/bin/activate   (Windows: .venv\\Scripts\\activate)"
install:    ; pip install -r requirements.txt
data:       ; $(PY) -m src.tbi_mlops.data.generate_synthetic
prepare:    ; $(PY) scripts/prepare_data.py
baseline:   ; $(PY) scripts/train_baseline.py
advanced:   ; $(PY) scripts/train_advanced.py
simulate:   ; $(PY) simulation/run_simulation.py
evaluate:   ; $(PY) scripts/evaluate.py
test:       ; $(PY) -m pytest -q 2>/dev/null || $(PY) scripts/run_tests.py
verify:     ; $(PY) scripts/verify_local.py
mlflow:     ; mlflow ui --backend-store-uri ./mlruns
api:        ; uvicorn src.tbi_mlops.serving.api:app --reload
dashboard:  ; streamlit run app.py
all: data prepare baseline advanced simulate evaluate test
