.PHONY: all clean build

CC      := gcc
CFLAGS  := -ansi -Wall -Wextra -Werror -pedantic-errors -Iinclude
LDFLAGS := -lm
SRC     := matrix_ops.c symnmf_algo.c symnmf.c
TARGET  := symnmf

all: build

build: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) $(LDFLAGS) -o $(TARGET)

clean:
	rm -f $(TARGET)
	find build -type f -name '*.o' -delete 2>/dev/null || true
