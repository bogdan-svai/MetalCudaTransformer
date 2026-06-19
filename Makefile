.PHONY: test check build verify install uninstall

test:
	python -m pytest -q

check:
	./mcu check examples/complex_app/app/main.cpp

build:
	./mcu build examples/complex_app/app/main.cpp

verify:
	./scripts/verify_stage2.sh

install:
	./scripts/install_mcu.sh

uninstall:
	./scripts/uninstall_mcu.sh
