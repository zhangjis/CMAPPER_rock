PYTHON=python3
PIP=pip3
CFLAGS="-Ofast -Wno-unreachable-code -Wno-unreachable-code-fallthrough"

all: dependencies build run

dependencies:
	@echo "Trying to install dependencies..."
	@${PYTHON} -m venv venv && source venv/bin/activate && ${PIP} install -r requirements.pip > /dev/null

build: heat_transport.pyx rocky_class.pyx
	@echo "Building Cython code..."
	@source venv/bin/activate && CFLAGS=${CFLAGS} ${PYTHON} setup_pre_adiabat.py build_ext --inplace

run: build
	@echo "Trying to run CMAPPER_rock"
	@source venv/bin/activate && ${PYTHON} test.py

clean:
	rm -rf venv build *.so *.c
