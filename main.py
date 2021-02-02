from random import *
from itertools import combinations
from math import log2, inf
from time import time


def XORsum(k, s):
    out = 0
    for i in s:
        out ^= k << i
    return out


def bitRep(e: int) -> str:
    return ''.join('{0:016b}'.format(e, 'b'))


def decode(encoded, k, s):
    return encoded ^ XORsum(k, s)


def sort(e):
    return len(e[0][1]) + len(e[1][1]) + len(e[2][1])


if __name__ == "__main__":
    L = []
    lengths = [8, 16]

    for k in lengths:
        L.append([])
        s0 = []
        for i in range(1, k + 2):
            s0.extend([list(n) for n in list(combinations(range(k + 1), i))])
        for _ in range(1000):
            # t0 = time()
            f1 = randint(0, 2 ** (2 * k) - 1)
            f2 = randint(0, 2 ** (2 * k) - 1)
            while f2 == f1:
                f2 = randint(0, 2 ** (2 * k) - 1)
            f3 = randint(0, 2 ** (2 * k) - 1)
            while f3 == f2 or f3 == f1:
                f3 = randint(0, 2 ** (2 * k) - 1)
            # f4 = randint(0, 2 ** (2 * k) - 1)
            # while f4 == f3 or f4 == f2 or f4 == f1:
            #     f4 = randint(0, 2 ** (2 * k) - 1)
            c12 = f1 ^ f2
            c13 = f1 ^ f3
            # c14 = f1 ^ f4
            c23 = f2 ^ f3
            # c24 = f2 ^ f4
            # c34 = f3 ^ f4

            possible = inf

            s0 = []
            for i in range(1, k + 2):
                s0.extend([list(n) for n in list(combinations(range(k + 1), i))])
            for p in s0:
                for q in s0:
                    if len(p) + len(q) < possible:
                        for r in s0:
                            if len(p) + len(q) + len(r) < possible:
                                # for s in s0:
                                #     if len(p) + len(q) + len(r) + len(s) < possible:
                                        for i in range(2 ** k):
                                            for n in range(2 ** k):
                                                if XORsum(i, p) ^ XORsum(n, q) == c12:
                                                    for m in range(2 ** k):
                                                        if XORsum(i, p) ^ XORsum(m, r) == c13 and XORsum(n, q) ^ XORsum(m, r) == c23:
                                                            possible = (len(p) + len(q) + len(r))
                                                            # for x in range(2 ** k):
                                                                # if (XORsum(x, s) ^ XORsum(i, p)) == c14 and (XORsum(x, s) ^ XORsum(n, q)) == c24 and (XORsum(x, s) ^ XORsum(m, r)) == c34:
                                                                    # possible.append([[i, p], [n, q], [m, r]])
                                                                    # possible = (len(p) + len(q) + len(r) + len(s))

            # print(f'{len(possible)} possible key pairings found')
            # possible.sort()
            L[-1].append(possible)
            # print(time() - t0)

        print(L[-1])

    with open('test.txt', 'w') as f:
        f.write("    " + "S".ljust(19) + "Delta B_s".ljust(23) + "Delta B".ljust(23) + "R_s".ljust(23) + "R".ljust(23) + "\n")
        f.write('EM1' + "\n")
        n = 3
        for i in range(len(L)):
            l = lengths[i]
            delta = (n * l) / (l - 1)
            S = [n < delta for n in L[i]]
            allSuccess = [d for d, s in zip(L[i], S) if s]
            p = [f'{2 * l} - {sum(S)}/{len(S)}={sum(S) / len(S)}' if len(S) != 0 else 'N/A',
                 f'{sum([i - n * l for i in allSuccess]) / len(allSuccess)}' if len(allSuccess) != 0 else 'N/A',
                 f'{sum([i - n * l for i in L[i]]) / len(L[i])}',
                 f'{round(sum([(i * log2(l) + n * l) / (2 * n * l) for i in allSuccess]) / len(allSuccess), 5)}' if len(
                     allSuccess) != 0 else 'N/A',
                 f'{round(sum([(i * log2(l) + n * l) / (2 * n * l) for i in L[i]]) / len(L[i]), 5)}']
            f.write(" | ".join([i.ljust(20) for i in p]) + "\n")
        f.write('EM2' + "\n")
        for i in range(len(L)):
            l = lengths[i]
            delta = (2 * n * l) / (l - 1)
            S = [n < delta for n in L[i]]
            allSuccess = [d for d, s in zip(L[i], S) if s]
            p = [f'{2 * l} - {sum(S)}/{len(S)}={sum(S) / len(S)}' if len(S) != 0 else 'N/A',
                 f'{sum([i - n * l for i in allSuccess]) / len(allSuccess)}' if len(allSuccess) != 0 else 'N/A',
                 f'{sum([i - n * l for i in L[i]]) / len(L[i])}',
                 f'{round(sum([(i * log2(l)) / (2 * n * l) for i in allSuccess]) / len(allSuccess), 5)}' if len(
                     allSuccess) != 0 else 'N/A',
                 f'{round(sum([(i * log2(l)) / (2 * n * l) for i in L[i]]) / len(L[i]), 5)}']
            f.write(" | ".join([i.ljust(20) for i in p]) + "\n")

    # just = 35
    # for i in possible:
    #     k1 = str(f'{bitRep(i[0][0])} {i[0][1]}').ljust(just)
    #     k2 = str(f'{bitRep(i[1][0])} {i[1][1]}').ljust(just)
    #     k3 = str(f'{bitRep(i[2][0])} {i[2][1]}').ljust(just)
    #     # k4 = str(f'{bitRep(i[3][0])} {i[3][1]}').ljust(just)
    #     f0 = f1 ^ XORsum(*i[0])
    #     f1d = f0 ^ XORsum(*i[0])
    #     f2d = f0 ^ XORsum(*i[1])
    #     f3d = f0 ^ XORsum(*i[2])
    #     # f4d = f0 ^ XORsum(*i[3])
    #     print(' | '.join([k1, k2, k3]) + ' || ' + f'{bitRep(f0)}' + ' || ' + ' | '.join([bitRep(n) for n in [f1d, f2d, f3d]]))
