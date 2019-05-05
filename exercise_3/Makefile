CC = mpicc
SRC = $(wildcard src/*.c)
OBJ = $(SRC:.c=.o)
BIN = ./bin
TARGET = pi.out

LDFLAGS = -fopenmp -std=gnu11 -g -Wall -lm -O2
CFLAGS = -I./include -fopenmp -std=gnu11 -g -Wall

all: dir $(BIN)/$(TARGET)

dir: ${BIN}

${BIN}:
	mkdir -p $(BIN)

$(BIN)/$(TARGET): $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS) $(CFLAGS)

.PHONY: clean
clean:
	rm -f $(OBJ) $(BIN)/$(TARGET)
