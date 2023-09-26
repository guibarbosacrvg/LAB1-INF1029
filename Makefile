CC = gcc
PROGRAM = matrix_lib_test

all: run

$(PROGRAM): matrix_lib.c timer.c matrix_lib_test.c
	$(CC) $(CFLAGS) -pthread -mfma -o $(PROGRAM) matrix_lib.c timer.c matrix_lib_test.c

.PHONY: clean run

clean:
	rm -f $(PROGRAM)

run: $(PROGRAM)
	./$(PROGRAM) 5.0 1024 1024 1024 1024 data/matrix_inputA_1024.dat data/matrix_inputB_1024.dat data/matrix_result1.dat data/matrix_result2.dat
	rm -f $(PROGRAM)
