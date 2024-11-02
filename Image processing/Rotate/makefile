# Nom de l'exécutable
EXEC = rotate

# Fichiers source
SRC = rotate.c

# Fichiers objets
OBJ = $(SRC:.c=.o)

# Options du compilateur
CFLAGS = -Wall -Wextra -O2

# Règle par défaut
all: $(EXEC)

# Règle pour créer l'exécutable
$(EXEC): $(OBJ)
	$(CC) $(OBJ) -o $(EXEC)

# Règle pour créer les fichiers objets
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Nettoyage des fichiers objets et de l'exécutable
clean:
	rm -f $(OBJ) $(EXEC)

# Nettoyage complet
distclean: clean

.PHONY: all clean distclean