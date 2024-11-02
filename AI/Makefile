CC = gcc
CFLAGS = -Wall -O2 -lm
TARGET = xor_ai
SRCS = xor.c
OBJS = ${SRCS:.c=.o}

${TARGET}: ${OBJS}
	${CC} ${CFLAGS} -o ${TARGET} ${OBJS} -lm


%.o: %.c
	${CC} ${CFLAGS} -c $< -o $@

clean:
	rm -f ${OBJS} ${TARGET}

run: ${TARGET}
	./${TARGET}