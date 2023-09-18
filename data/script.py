import numpy as np

if __name__ == "__main__":
    array_2048A = np.full((2048, 2048), 3.0, dtype="float32")
    array_2048B = np.full((2048, 2048), 2.0, dtype="float32")
    array_2048A.tofile("data/matrix_inputA_2048.dat")
    array_2048B.tofile("data/matrix_inputB_2048.dat")
