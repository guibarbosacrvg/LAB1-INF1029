import numpy as np

if __name__ == "__main__":
    array_256A = np.full((256, 256), 17.0, dtype="float32")
    array_256B = np.full((256, 256), 15.0, dtype="float32")
    array_256A.tofile("matrix_inputA_256.dat")
    array_256B.tofile("matrix_inputB_256.dat")
