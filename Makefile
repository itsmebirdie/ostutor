.PHONY: install lint types test cov check demo eval eval-rag api ui clean

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

# real bge evaluation on YOUR corpus and questions; see docs/RAG_EVAL.md
eval-rag:
	python eval/run_rag_real.py --corpus data/corpus_raw --questions eval/my_questions.jsonl --embedder bge-small

api:
	uvicorn ostutor.api.app:app --reload

ui:
	streamlit run streamlit_app.py

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml
