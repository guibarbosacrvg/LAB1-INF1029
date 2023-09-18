CC = gcc
PROGRAM = matrix_lib_test

all: run

$(PROGRAM): matrix_lib.c timer.c matrix_lib_test.c
	$(CC) $(CFLAGS) -mfma -o $(PROGRAM) matrix_lib.c timer.c matrix_lib_test.c

.PHONY: clean run

clean:
	rm -f $(PROGRAM)

run: $(PROGRAM)
	./$(PROGRAM) 5.0 2048 2048 2048 2048 data/matrix_inputA_2048.dat data/matrix_inputB_2048.dat data/matrix_result1.dat data/matrix_result2.dat
	rm -f $(PROGRAM)
