CC = nvcc
PROGRAM = matrix_lib_test

all: run

$(PROGRAM): matrix_lib.cu timer.c matrix_lib_test.cu
	$(CC) $(CFLAGS) -o $(PROGRAM) matrix_lib.cu timer.c matrix_lib_test.cu

.PHONY: clean run

clean:
	rm -f $(PROGRAM)

run: $(PROGRAM)
	./$(PROGRAM) 5.0 1024 1024 1024 1024 8 ../data/matrix_inputA_1024.dat ../data/matrix_inputB_1024.dat ../data/matrix_result1.dat ../data/matrix_result2.dat
	rm -f $(PROGRAM)

build:
	$(CC) $(CFLAGS) -o $(PROGRAM) matrix_lib.cu timer.c matrix_lib_test.cu

2048:
	./$(PROGRAM) 5.0 2048 2048 2048 2048 256 4096 1024 ../data/matrix_inputA_2048.dat ../data/matrix_inputB_2048.dat ../data/matrix_result1.dat
	
512:
	./$(PROGRAM) 5.0 512 512 512 512 8 ../data/matrix_inputA_512.dat ../data/matrix_inputB_512.dat ../data/matrix_result1.dat ../data/matrix_result2.dat
	rm -f $(PROGRAM)

debug: $(PROGRAM)
	gdb $(PROGRAM)