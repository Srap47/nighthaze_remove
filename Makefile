.PHONY: install dev test lint clean

install:
	cd backend && uv venv --python 3.11 && uv pip install -r requirements.txt
	cd frontend && npm install

dev:
	@echo "Starting backend on :8000 and frontend on :5173"

test:
	cd backend && .venv\Scripts\activate && pytest tests/ -v --cov=app --cov-report=term-missing

lint:
	cd backend && black app/ && isort app/

clean:
	for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
