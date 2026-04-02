.PHONY: dev test test-unit test-integration lint index eval

dev:
	uvicorn rag_assistant.api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ --cov=src --cov-report=term-missing -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v -m integration

lint:
	ruff check src/ tests/
	mypy src/

index:
	@test -n "$(URL)" || (echo "Usage: make index URL=<repo_url>" && exit 1)
	curl -s -X POST http://localhost:8000/admin/index \
		-H "Content-Type: application/json" \
		-d '{"repo_url": "$(URL)"}' | python -m json.tool

query:
	@test -n "$(Q)" || (echo "Usage: make query Q='<question>'" && exit 1)
	curl -s -X POST http://localhost:8000/query \
		-H "Content-Type: application/json" \
		-d '{"query": "$(Q)"}' | python -m json.tool

eval:
	python eval/run_eval.py --dataset eval/dataset.jsonl
