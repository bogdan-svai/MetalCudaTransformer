.PHONY: test lint lint-full format-check qa check build verify cuda-check install-hooks install uninstall

test:
	python -m pytest -q

lint:
	python -m black --check core/passes/metal_optimizer.py core/passes/ir_optimization.py core/frontend/semantic.py core/backends/device.py core/backends/metal.py core/ir/symbols.py
	python -m isort --check-only core/passes/metal_optimizer.py core/passes/ir_optimization.py core/frontend/semantic.py core/backends/device.py core/backends/metal.py core/ir/symbols.py

lint-full:
	python -m black --check .
	python -m isort --check-only .

format-check: lint

qa:
	python scripts/qa_complexity.py

check:
	./mcu check examples/complex_app/app/main.cpp

build:
	./mcu build examples/complex_app/app/main.cpp

verify:
	./scripts/verify.sh

cuda-check:
	python scripts/cuda_checker.py --tier all

install-hooks:
	git config core.hooksPath .githooks
	chmod +x .githooks/pre-push

install:
	./install.sh

uninstall:
	./scripts/uninstall_mcu.sh
