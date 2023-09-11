import numpy as np

if __name__ == "__main__":
    array_1024A = np.full((1024, 1024), 17.0, dtype="float32")
    array_1024B = np.full((1024, 1024), 15.0, dtype="float32")
    array_1024A.tofile("matrix_inputA_1024.dat")
    array_1024B.tofile("matrix_inputB_1024.dat")
