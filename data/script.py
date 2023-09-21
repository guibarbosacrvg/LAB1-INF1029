import numpy as np

if __name__ == "__main__":
    dimmension = (int(input("Type the dimmensions to be used on the matrix: ")))
    array_dimmensionA = np.full((dimmension, dimmension), 3.0, dtype="float32")
    array_dimmensionB = np.full((dimmension, dimmension), 2.0, dtype="float32")
    array_dimmensionA.tofile(f"data/matrix_inputA_{dimmension}.dat")
    array_dimmensionB.tofile(f"data/matrix_inputB_{dimmension}.dat")
