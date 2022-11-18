style:
	python -m black --line-length 119 --target-version py39 .
	python -m isort .

quality:
	python -m black --check --line-length 119 --target-version py39 .
	python -m isort --check-only .
	python -m flake8 --max-line-length 119
	python -m mypy ./benchmarks

typecheck-benchmarks:
	python -m mypy ./benchmarks

test:
	python -m pytest -sv tests/