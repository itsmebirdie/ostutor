.PHONY: install lint types test cov check demo eval api ui clean

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check .

types:
	mypy

test:
	pytest -q

cov:
	pytest -q --cov --cov-report=term-missing

# Everything CI runs, in one command
check: lint types cov

demo:
	python -m ostutor.slice.demo --all --tamper

eval:
	python eval/run_all.py

api:
	uvicorn ostutor.api.app:app --reload

ui:
	streamlit run streamlit_app.py

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml
