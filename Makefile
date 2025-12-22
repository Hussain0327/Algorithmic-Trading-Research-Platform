.PHONY: setup demo dashboard test report clean help

help:
	@echo "Trading Strategy Validation - Available Commands"
	@echo ""
	@echo "  make setup      Create venv and install dependencies"
	@echo "  make demo       Run CLI backtest demo"
	@echo "  make dashboard  Launch Streamlit dashboard"
	@echo "  make test       Run pytest test suite"
	@echo "  make report     Generate HTML backtest report"
	@echo "  make clean      Remove venv and cache files"

setup:
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	@echo ""
	@echo "Setup complete. Run 'make demo' to test."

demo:
	./venv/bin/python main.py

dashboard:
	./venv/bin/streamlit run app.py

test:
	./venv/bin/pytest -v

report:
	./venv/bin/python -m analytics.report

clean:
	rm -rf venv __pycache__ */__pycache__ .pytest_cache reports/*.html
	@echo "Cleaned up generated files."
