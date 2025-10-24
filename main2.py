import numpy as np
import timeit
import matplotlib.pyplot as plt

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


# C = np.array([[7,-5,2,-0.5,4],
# [-8,-7,0.5,-3,5],
# [2,8,-2,-4,-6],
# [6,-0.5,-6,-0.5,4],
# [-6,-5,8 ,7 ,0.5]], dtype=float)

# X = np.array([0, 6, -4, -3, -1])

# print(f"uklad rownan: ")
# for i in range(C.shape[0]):
#     for j in range(C.shape[1]):
#         print(f"{C[i][j]}x{j+1}", end=" ")
#         if j is not C.shape[1]-1:
#             print('',end='+ ')
#     print(f"= {X[i]}")

# print()
# print("wynik: ")
# B = metoda_eliminacji_gaussa(C, X)

# for i in range(B.shape[0]):
#     print(f"x{i+1} = {B[i]:.10f}", end="\t")
# print()

# print()
# print("wynik z funcji wbudowanych: ")
# B2 = np.linalg.solve(C, X)
# for i in range(B2.shape[0]):
#     print(f"x{i+1} = {B2[i]:.10f}", end="\t")
# print()


# print()
# print("roznica miedzy wynikami: ")
# B2 = np.linalg.solve(C, X)
# for i in range(B2.shape[0]):
#     print(f"x{i+1} = {np.abs(B[i]-B2[i]):.10f}", end="\t")
# print()

def metoda_thomasa(A, B):
    n = A.shape[0]

    a = np.zeros(n)
    a[1:] = np.diag(A, k=-1)

    b = np.diag(A)

    c = np.zeros(n)
    c[:-1] = np.diag(A, k=1)

    d = np.copy(B)

    bet = np.zeros(n)
    gam = np.zeros(n)

    bet[0] = -c[0] / b[0]
    gam[0] = d[0] / b[0]

    for i in range(1, n):
        bet[i] = -c[i] / (a[i] * bet[i-1] + b[i])
        gam[i] = (d[i] - a[i] * gam[i-1]) / (a[i] * bet[i-1] + b[i])

    x = np.zeros(n)
    x[n-1] = gam[n-1]
    
    for i in range(n-2, -1, -1):
        x[i] = bet[i] * x[i+1] + gam[i]

    return x

print("zad 1")
print('\n')

A = np.array([
    [7, 0.5, 0, 0, 0],
    [-5, -2, -2, 0, 0],
    [0, -8, -8, -3, 0],
    [0, 0, -4, -8, -2],
    [0, 0, 0, 2, -2]
], dtype=float)

B = np.array([6, -0.5, 4, -0.5, 6], dtype=float)

t = metoda_thomasa(A, B)
t1 = np.linalg.solve(A, B)

print("metoda thomasa: ")
for i in range(t.shape[0]):
    print(f"x{i+1} = {t[i]:.10f}", end="\t")
print('\n')

print("linalg.solve: ")
for i in range(t1.shape[0]):
    print(f"x{i+1} = {t1[i]:.10f}", end="\t")
print('\n')

print("roznica")
for i in range(t.shape[0]):
    print(f"x{i+1} = {np.abs(t[i]-t1[i]):.10f}", end="\t")
print('\n')


print()
print('*' * 40)
print("zad 2")
print()

def losuj_gauss(low, high, n):
    A = np.random.rand(n, n) * (high - low) + low
    B = np.random.rand(n) * (high - low) + low
    return A,B

def losuj_thomas(low, high, n):
    A = np.zeros((n,n), dtype=float)
    for i in range(n):
        if i - 1 >= 0:
            A[i][i-1] = (np.random.rand() * (high - low) + low)
        if i + 1 < n:
            A[i][i+1] = (np.random.rand() * (high - low) + low)
        A[i][i] = (np.random.rand() * (high - low) + low)

    B = np.random.rand(n) * (high - low) + low

    return A, B


print("wylosowana macierz dla metody gaussa: \n")  
rand_A, rand_B = losuj_gauss(-3, 10, 5)
print(f"rand A:\n{rand_A}")
print()
print(f"rand B:\n{rand_B}")
print("\n")

g = metoda_eliminacji_gaussa(rand_A, rand_B)
print("wynik metody gaussa: ")
for i in range(g.shape[0]):
    print(f"x{i+1} = {g[i]:.10f}", end="\t")
print()

g1 = np.linalg.solve(rand_A, rand_B)
print()
print("wynik z funcji wbudowanych: ")
for i in range(g1.shape[0]):
    print(f"x{i+1} = {g1[i]:.10f}", end="\t")
print()


print()
print("roznica miedzy wynikami: ")
for i in range(g1.shape[0]):
    print(f"x{i+1} = {np.abs(g[i]-g1[i]):.10f}", end="\t")
print()

print("\n\n")


print("wylosowana macierz dla metody thomasa: \n")  
rand_A, rand_B = losuj_thomas(-9, 6, 4)
print(f"rand A:\n{rand_A}")
print()
print(f"rand B:\n{rand_B}")
print("\n")

t = metoda_thomasa(rand_A, rand_B)
print("wynik metody thomasa: ")
for i in range(t.shape[0]):
    print(f"x{i+1} = {t[i]:.10f}", end="\t")
print()

t1 = np.linalg.solve(rand_A, rand_B)
print()
print("wynik z funcji wbudowanych: ")
for i in range(t1.shape[0]):
    print(f"x{i+1} = {t1[i]:.10f}", end="\t")
print()


print()
print("roznica miedzy wynikami: ")
for i in range(t.shape[0]):
    print(f"x{i+1} = {np.abs(t[i]-t1[i]):.10f}", end="\t")
print()



print("\n")
print('*' * 40)
print("zad 3")
print("CZAS OBLICZEŃ")
print()

k = 500
gauss_time = np.zeros(24)
thomas_time = np.zeros(24)
numpy_time = np.zeros(24)

for n in range(2, 26):
    A, B = losuj_thomas(-5, 7, n)

    print(f"porownanie dla n={n}:")

    print('-' * 40)
    
    a,b = np.copy(A), np.copy(B)
    gauss_time[n-2] = timeit.timeit(lambda: metoda_eliminacji_gaussa(a,b) , number=k)
    print(f"gauss:\t{gauss_time[n-2]:.5f}")

    a,b = np.copy(A), np.copy(B)
    thomas_time[n-2] = timeit.timeit(lambda: metoda_thomasa(a,b) , number=k)
    print(f"thomas:\t{thomas_time[n-2]:.5f}")

    a,b = np.copy(A), np.copy(B)
    numpy_time[n-2] = timeit.timeit(lambda: np.linalg.solve(a,b) , number=k)
    print(f"numpy:\t{numpy_time[n-2]:.5f}")

    print('-' * 40)

print(gauss_time)
plt.figure()
plt.scatter(range(2, 26), gauss_time / thomas_time)
plt.xlabel('n')
plt.ylabel('time [s]')
plt.title('czas obliczeń - gauss/thomas')
plt.show()

plt.figure()
plt.scatter(range(2, 26), numpy_time / thomas_time)
plt.xlabel('n')
plt.ylabel('time [s]')
plt.title('czas obliczeń - numpy/thomas')
plt.show()


