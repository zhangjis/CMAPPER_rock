PYTHON=python3
PIP=pip3
CFLAGS="-Ofast -Wno-unreachable-code -Wno-unreachable-code-fallthrough"

all: dependencies build run

dependencies:
	@echo "\n**********************************************"\
		"MAKEFILE: Trying to install dependencies..."\
		"\n**********************************************\n"
	@${PYTHON} -m venv venv && source venv/bin/activate && ${PIP} install -r requirements.pip > /dev/null

build: heat_transport.pyx rocky_class.pyx
	@echo "\n**********************************************"\
		"\nMAKEFILE: Building Cython code..."\
		"\n**********************************************\n"
	@source venv/bin/activate && CFLAGS=${CFLAGS} ${PYTHON} setup_pre_adiabat.py build_ext --inplace

run: build
	@echo "\n**********************************************"\
		"\nMAKEFILE: Trying to run CMAPPER_rock"\
		"\n************************************************\n"
	@source venv/bin/activate && ${PYTHON} test.py

clean:
	rm -rf venv build *.so *.c
