CC = gcc
PROGRAM = matrix_lib_test

all: run

$(PROGRAM): matrix_lib.c timer.c matrix_lib_test.c
	$(CC) $(CFLAGS) -std=c11 -pthread -mfma -o $(PROGRAM) matrix_lib.c timer.c matrix_lib_test.c

.PHONY: clean run

clean:
	rm -f $(PROGRAM)

run: $(PROGRAM)
	./$(PROGRAM) 5.0 1024 1024 1024 1024 8 data/matrix_inputA_1024.dat data/matrix_inputB_1024.dat data/matrix_result1.dat data/matrix_result2.dat
	rm -f $(PROGRAM)

build:
	$(CC) $(CFLAGS) -std=c11 -pthread -mfma -o $(PROGRAM) matrix_lib.c timer.c matrix_lib_test.c

2048:
	./$(PROGRAM) 5.0 2048 2048 2048 2048 8 data/matrix_inputA_2048.dat data/matrix_inputB_2048.dat data/matrix_result1.dat data/matrix_result2.dat
	rm -f $(PROGRAM)

512:
	./$(PROGRAM) 5.0 512 512 512 512 8 data/matrix_inputA_512.dat data/matrix_inputB_512.dat data/matrix_result1.dat data/matrix_result2.dat
	rm -f $(PROGRAM)
