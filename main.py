import numpy as np

# np.linalg.solve(A, B)

def metoda_eliminacji_gaussa(A, X):

    n = A.shape[0]

    C = np.zeros((A.shape[0], A.shape[1]+1), dtype=float)
    for i in range(0, n):
        for j in range(0, n):
            C[i][j] = A[i][j]
        C[i][n] = X[i]

    X = np.zeros(n, dtype=float)

    for s in range(0, n-1):
        for i in range(s+1, n):
            for j in range(s+1, n+1):
                C[i][j] = C[i][j] - (C[i][s] / C[s][s]) * C[s][j]

    X[n-1] = C[n-1][n] / C[n-1][n-1]
    for i in range(n-2, -1, -1):
        suma = 0
        for s in range(i+1, n):
            suma += C[i][s] * X[s]

        X[i] = (C[i][n] - suma) / C[i][i]

    return X






# 7x1-5x2 +2x3-0.5x4 +4x5 = 0
# -8x1-7x2 +0.5x3-3x4 +5x5 = 6
# 2x1 +8x2-2x3-4x4-6x5 = -4
# 6x1-0.5x2-6x3-0.5x4 +4x5 = -3
# -6x1-5x2 +8x3 +7x4 +0.5x5 = -1



C = np.array([[7,-5,2,-0.5,4],
[-8,-7,0.5,-3,5],
[2,8,-2,-4,-6],
[6,-0.5,-6,-0.5,4],
[-6,-5,8 ,7 ,0.5]], dtype=float)

X = np.array([0, 6, -4, -3, -1])

print(f"uklad rownan: ")
for i in range(C.shape[0]):
    for j in range(C.shape[1]):
        print(f"{C[i][j]}x{j+1}", end=" ")
        if j is not C.shape[1]-1:
            print('',end='+ ')
    print(f"= {X[i]}")

print()
print("wynik: ")
B = metoda_eliminacji_gaussa(C, X)

for i in range(B.shape[0]):
    print(f"x{i+1} = {B[i]:.10f}", end="\t")
print()

print()
print("wynik z funcji wbudowanych: ")
B2 = np.linalg.solve(C, X)
for i in range(B2.shape[0]):
    print(f"x{i+1} = {B2[i]:.10f}", end="\t")
print()


print()
print("roznica miedzy wynikami: ")
B2 = np.linalg.solve(C, X)
for i in range(B2.shape[0]):
    print(f"x{i+1} = {np.abs(B[i]-B2[i]):.10f}", end="\t")
print()




