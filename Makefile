PYTHON=python3
PIP=pip3

all: install build run

install:
	@echo "Trying to install dependencies..."
	@${PYTHON} -m venv venv && source venv/bin/activate && ${PIP} install -r requirements.pip

build:
	@echo "Building Cython code..."
	@source venv/bin/activate && ${PYTHON} setup_pre_adiabat.py build_ext --inplace

run:
	@echo "Trying to run CMAPPER_rock"
	@source venv/bin/activate && ${PYTHON} test.py

clean:
	rm -rf venv build *.so *.c
