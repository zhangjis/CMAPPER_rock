PYTHON=python3
PIP=pip3
CFLAGS="-Ofast -Wno-unreachable-code -Wno-unreachable-code-fallthrough"

all: check-python check-conda dependencies build run

check-python:
	@echo "\n**********************************************"\
		"\nChecking Python version..."\
		"\n**********************************************\n"
	@${PYTHON} --version > /dev/null 2>&1\
		&& echo "Note: Python version is $$(${PYTHON} --version) and pip version is $$(${PIP} --version | cut -d ' ' -f1-2). Looks like it should be good!"\
		|| ( echo "Python install not found, please install ${PYTHON}" && exit 1 )

check-conda:
ifeq ($(CMAP_CONDA_CHECK), FALSE)
	@echo "Skipping Anaconda \$$PATH check..."
else
	@if [ $$(which -a ${PYTHON} | grep "opt/anaconda*/envs/*/bin/${PYTHON}" | wc -l) -ge 2 ]; then\
		echo "Multiple Anaconda envs found in \$$PATH, exiting. Run \`export \$$CMAP_CONDA_CHECK=FALSE\` to disable this check" && exit 1;\
	fi
endif

dependencies:
	@echo "\n**********************************************"\
		"\nMAKEFILE: Trying to install dependencies..."\
		"\n**********************************************\n"
	@${PYTHON} -m venv .venv && source .venv/bin/activate && ${PIP} install -r requirements.pip

.PHONY: build
build: heat_transport.pyx rocky_class.pyx
	@echo "\n**********************************************"\
		"\nMAKEFILE: Building Cython code..."\
		"\n**********************************************\n"
	@source .venv/bin/activate && CFLAGS=${CFLAGS} ${PYTHON} setup_pre_adiabat.py build_ext --inplace

run: build
	@echo "\n**********************************************"\
		"\nMAKEFILE: Trying to run CMAPPER_rock"\
		"\n************************************************\n"
	@source .venv/bin/activate && ${PYTHON} test.py

clean:
	rm -rf venv .venv build *.so *.c
