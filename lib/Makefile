CC = gcc
CFLAGS = -g -Wall -fopenmp -O3

OPENBLAS_PATH = /opt/OpenBLAS

OBJECTS = NN_utils.o NN_math.o NN_core.o AE.o
INCFLAGS = -I $(OPENBLAS_PATH)/include -I ../include
LDFLAGS = -L$(OPENBLAS_PATH)/lib -L../lib
#LIBS = -lopenblas -lcsparse -fopenmp
LIB_TARGET = libNN.a

RANLIB = ranlib

all: libNN.a

libNN.a: $(OBJECTS) 
	rm -f $(LIB_TARGET)
	$(AR) $(ARFLAGS) $(LIB_TARGET) $(OBJECTS)
	- $(RANLIB) $(LIB_TARGET)

%.o : ../src/%.c ../include/%.h
	$(CC) -o $@ -c $(CFLAGS) $< $(INCFLAGS)

.SUFFIXES:
.SUFFIXES:  .c .cc .C .cpp .o


clean:
	rm -f *.o $(LIB_TARGET) 

.PHONY: all
.PHONY: count
.PHONY: clean
