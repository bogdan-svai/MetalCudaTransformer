.PHONY: test check build verify cuda-check install-hooks install uninstall

test:
	python -m pytest -q

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
	./scripts/install_mcu.sh

uninstall:
	./scripts/uninstall_mcu.sh
