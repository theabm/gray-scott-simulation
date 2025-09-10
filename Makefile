# =======================
# Project-wide Makefile
# =======================

# ---- C config ----
CC        := mpicc
CFLAGS    := -O3 -std=c11 -Wall -Wextra 
C_SRC     := c/src/sim.c
C_BIN     := c/sim

# ---- Python config ----
PY_VENV   := .doreisa-env
PY        := $(PY_VENV)/bin/python
PIP       := $(PY_VENV)/bin/pip
PY_MAIN   := python/sim.py

# ---- MPI + demo args ----
MPIRUN    := mpirun
NP        := 4

RUN_ARGS_C  := --steps 20000 --print-every 200 --seed-mode local --periodic --viz-every 50 --viz-gif --viz-outdir c/frames
RUN_ARGS_PY := --steps 20000 --print-every 200 --seed-mode local --periodic --viz-every 50 --viz-gif --viz-outdir python/frames

# ---- Tools ----
MAGICK    := magick

# ---- Phony targets ----
.PHONY: help all \
        c-build c-run c-gif c-clean \
        py-venv py-install py-run py-gif py-clean \
        demo clean

help:
	@echo "Targets:"
	@echo "  c-build      - build C MPI sim"
	@echo "  c-run        - run C sim with preset args"
	@echo "  c-gif        - GIF from c/frames/*.pgm"
	@echo "  c-clean      - remove C binary and c/frames"
	@echo "  py-venv      - create .doreisa-env if missing"
	@echo "  py-install   - pip install python/requirements.txt (if present)"
	@echo "  py-run       - run Python sim with preset args (no activation needed)"
	@echo "  py-gif       - GIF from python/frames/*.png"
	@echo "  demo         - run both C and Python demos"
	@echo "  clean        - clean both C and Python outputs"

all: c-build

# ===== C targets =====
c-build: $(C_BIN)

$(C_BIN): $(C_SRC)
	$(CC) $(CFLAGS) -o $(C_BIN) $(C_SRC)

c-run: c-build
	$(MPIRUN) -n $(NP) ./$(C_BIN) $(RUN_ARGS_C)
	$(MAGICK) c/frames/step_*.pgm c/frames/out.gif

c-clean:
	rm -f $(C_BIN)
	rm -rf c/frames

# ===== Python targets =====
py-venv:
	test -d $(PY_VENV) || python3 -m venv $(PY_VENV)

py-install: py-venv
	@if [ -f python/requirements.txt ]; then \
	  $(PIP) install -U pip && $(PIP) install -r python/requirements.txt ; \
	else \
	  echo "No python/requirements.txt found; skipping installs."; \
	fi

# python version already creates the GIF if --viz-gif is given
py-run: py-venv
	$(MPIRUN) -n $(NP) $(PY) $(PY_MAIN) $(RUN_ARGS_PY)

py-clean:
	rm -rf python/frames

# ===== Convenience =====
demo: c-run py-run

clean: c-clean py-clean
