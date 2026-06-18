import numpy as np
import matplotlib.pyplot as plt
from utils import plot_lines

a=np.array([
    [-1,3],
    [3,2]
], dtype=np.dtype(float))

b=np.array([7, 1], dtype=np.dtype(float))

print("Matrix a:")
print(a)
print("Vector b:")
print(b)


print(np.shape(a))
print(np.shape(b))

x=np.linalg.solve(a,b)
print("Solution x:")
print(x)






### DETERMINANT EVALUATION ####

d=np.linalg.det(a)
print("Determinant of a:")
print(f"{d:.2f}")



#### REPRESENTATION OF THE SYSTEM OF EQUATIONS ####

 #### BEFORE VISUALIZATIION CONVERT INOT MATRIX, and here b is not in 2-d so we need to convert it into 2-d array


A_SYSTEM=np.hstack((a,b.reshape(2,1)))
print("Augmented matrix [A|b]:")
print(A_SYSTEM)
print(A_SYSTEM[1])




# graph os linear equations

plot_lines(A_SYSTEM)


#### Linear Equations with No Solutions


A_2 = np.array([
        [-1, 3],
        [3, -9]
    ], dtype=np.dtype(float))

b_2 = np.array([7, 1], dtype=np.dtype(float))

d_2 = np.linalg.det(A_2)

try:
    x_2 = np.linalg.solve(A_2, b_2)
except np.linalg.LinAlgError as err:
    print(err)
print("Determinant of A_2:")
print(f"{d_2:.2f}")


A2_SYSTEM=np.hstack((A_2,b_2.reshape(2,1)))
print("Augmented matrix [A_2|b_2]:")
print(A2_SYSTEM)
plot_lines(A2_SYSTEM)


# Linear Equations with Infinitely Many Solutions

A_3 = np.array([
        [-1, 3],
        [3, -9]
    ], dtype=np.dtype(float))

b_3 = np.array([7, -21], dtype=np.dtype(float))
d_3 = np.linalg.det(A_3)
try:
    x_3 = np.linalg.solve(A_3, b_3)
except np.linalg.LinAlgError as err:
    print(err)

print("Determinant of A_3:")
print(f"{d_3:.2f}")
A3_SYSTEM=np.hstack((A_3,b_3.reshape(2,1)))
print("Augmented matrix [A_3|b_3]:")
print(A3_SYSTEM)
plot_lines(A3_SYSTEM)

