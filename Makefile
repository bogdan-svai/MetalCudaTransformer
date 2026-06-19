.PHONY: test check build verify install-hooks install uninstall

test:
	python -m pytest -q

check:
	./mcu check examples/complex_app/app/main.cpp

build:
	./mcu build examples/complex_app/app/main.cpp

verify:
	./scripts/verify.sh

install-hooks:
	git config core.hooksPath .githooks
	chmod +x .githooks/pre-push

install:
	./scripts/install_mcu.sh

uninstall:
	./scripts/uninstall_mcu.sh
