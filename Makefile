CC = gcc
CFLAGS = -g -Wall -fopenmp -O3

CSPARSE_PATH = CSparse
OPENBLAS_PATH = /opt/OpenBLAS

OBJECTS = NN_utils.o NN_math.o NN_core.o AE.o train_AE.o
INCFLAGS = -I $(OPENBLAS_PATH)/include -I $(CSPARSE_PATH)/Include
LDFLAGS = -L$(OPENBLAS_PATH)/lib -L$(CSPARSE_PATH)/Lib
LIBS = -lopenblas -lcsparse -fopenmp
TARGET = trainae

all: train_ae

train_ae: $(OBJECTS)
	$(CC) -o $(TARGET) $(OBJECTS) $(LDFLAGS) $(LIBS)

.SUFFIXES:
.SUFFIXES:	.c .cc .C .cpp .o

.c.o :
	$(CC) -o $@ -c $(CFLAGS) $< $(INCFLAGS)

count:
	wc *.c *.cc *.C *.cpp *.h *.hpp

clean:
	rm -f *.o

.PHONY: all
.PHONY: count
.PHONY: clean
