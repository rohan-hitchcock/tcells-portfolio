# template makefile

CC     = gcc
CFLAGS = -Wall -fopenmp
# modify the flags here ^

EXE    = ../tcell_sim_old
OBJ    = main.o graph.o sim.o pcg_basic.o
# add object files here ^

# top (default) target
all: $(EXE)

# how to link executable
$(EXE): $(OBJ)
	$(CC) $(CFLAGS) -o $(EXE) $(OBJ)

# dependencies
main.o: graph.h
sim.o: graph.h sim.h pcg_basic.h
graph.o: graph.h

# phony targets (these targets do not represent actual files)
.PHONY: clean cleanly all CLEAN

# `make clean` to remove all object files
# `make CLEAN` to remove all object and executable files
# `make cleanly` to `make` then immediately remove object files (inefficient)
clean:
	rm -f $(OBJ)
CLEAN: clean
	rm -f $(EXE)
cleanly: all clean
