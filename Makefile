
## USER SPECIFIC DIRECTORIES ##

# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda-8.0


## CC COMPILER OPTIONS ##

# CC compiler options:
CC=g++
CC_FLAGS=-std=c++11
CC_LIBS=

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS=-std=c++11
NVCC_LIBS=

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart -lcublas

## Project file structure ##

# Source file directory:
SRC_DIR = src

# Object file directory:
OBJ_DIR = bin

# Include header file diretory:
INC_DIR = include


## Make variables ##

# Target executable name:
EXE = run_test

OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/solver.o

## Compile ##

# Link c++ and CUDA compiled object files to target executable:
run_test : $(OBJS)
	$(CC) $(CC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)


$(OBJ_DIR)/main.o : main.cpp
	$(CC) $(CC_FLAGS) -c $< -o $@ -I ${INC_DIR}

# Compile CUDA source files to object files:
$(OBJ_DIR)/solver.o : ${SRC_DIR}/solver.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS) -I ${INC_DIR}

# Clean objects in object directory.
clean:
	$(RM) bin/* *.o $(EXE)