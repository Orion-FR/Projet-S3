CC = gcc
CFLAGS = -Wall -Wextra -O2
LIBS = -lm -lSDL2 -lSDL2_image
SRC = main.c neural.c network.c
OBJ = $(SRC:.c=.o)
EXEC = neural_net

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)


%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@


clean:
	rm -f $(OBJ) $(EXEC)

.PHONY: all clean