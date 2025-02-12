PYTHON=python3
PIP=pip3
CFLAGS="-Ofast -Wno-unreachable-code -Wno-unreachable-code-fallthrough"

all: check-python check-conda dependencies build run

check-python:
	@printf "%b\n" "\n**********************************************"\
		"Checking Python version..."\
		"**********************************************\n"
	@${PYTHON} --version > /dev/null 2>&1\
		&& printf "%b\n" "Note: Python version is $$(${PYTHON} --version) and pip version is $$(${PIP} --version | cut -d ' ' -f1-2). Looks like it should be good!\n"\
		|| ( printf "%b\n" "Python install not found, please install ${PYTHON}\n" && exit 1 )

check-conda:
ifeq ($(CMAP_CONDA_CHECK), FALSE)
	@printf "%b\n" "Skipping Anaconda \$$PATH check..."
else
	@if [ $$(which -a ${PYTHON} | grep "opt/anaconda*/envs/*/bin/${PYTHON}" | wc -l) -ge 2 ]; then\
		printf "%b\n" "Multiple Anaconda envs found in \$$PATH, exiting. Run \`export \$$CMAP_CONDA_CHECK=FALSE\` to disable this check" && exit 1;\
	fi
endif

dependencies:
	@printf "%b\n" "\n**********************************************"\
		"MAKEFILE: Trying to install dependencies..."\
		"**********************************************\n"
	@${PYTHON} -m venv .venv && source .venv/bin/activate && ${PIP} install -r requirements.pip


.PHONY: build
build: mie_gruneisen_debye.pyx
	@printf "%b\n" "\n**********************************************"\
		"MAKEFILE: Building Cython code..."\
		"**********************************************\n"
	@source .venv/bin/activate && CFLAGS=${CFLAGS} ${PYTHON} setup.py build_ext --inplace

run: build
	@printf "%b\n" "\n**********************************************"\
		"MAKEFILE: Trying to compute EoS table for Fe16Si"\
		"************************************************\n"
	@source .venv/bin/activate && ${PYTHON} run.py

# removes all python packages and compiled Cython code
clean:
	rm -rf venv .venv build *.so *.c

# removes everything, including results folders
clean-all:
	git clean -fdx
