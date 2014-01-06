CC = gcc
CFLAGS = -g -Wall -fopenmp -O3

OPENBLAS_PATH = /opt/OpenBLAS

INCFLAGS = -I$(OPENBLAS_PATH)/include -I../include
LDFLAGS = -L$(OPENBLAS_PATH)/lib -L../lib
LIBS = -lNN -lopenblas -lcsparse -fopenmp
TARGET = trainae

all: train_ae

train_ae: train_AE.o
	$(CC) -o $(TARGET) train_AE.o $(LDFLAGS) $(LIBS)

.SUFFIXES:
.SUFFIXES:  .c .cc .C .cpp .o

.c.o :
	$(CC) -o $@ -c $(CFLAGS) $< $(INCFLAGS)

count:
	wc *.c *.cc *.C *.cpp *.h *.hpp

clean:
	rm -f *.o

.PHONY: all
.PHONY: count
.PHONY: clean
