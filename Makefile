.PHONY: test check install uninstall

test:
	python -m pytest -q

check:
	./mcu check examples/complex_app/app/main.cpp

install:
	./scripts/install_mcu.sh

uninstall:
	./scripts/uninstall_mcu.sh
