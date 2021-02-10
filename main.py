from random import *
from itertools import combinations
from math import log2, inf
from time import time


class Item:
    def __init__(self, iterable, child=None):
        self.value = 0
        self.child = child
        self.currentIteration = None
        self.iterable = iterable

    def iterate(self, iteration):
        if n > iteration > 0:
            # BSL, not first
            for i_ in self.iterable:
                self.currentIteration = i_
                curLength = sum([len(p.currentIteration) for p in possible[:iteration + 1]])
                # Length check
                if curLength < possible[-1].child:
                    self.child.iterate(iteration + 1)
                else:
                    break
        elif iteration > n:
            # Key, not first
            for i_ in self.iterable:
                self.currentIteration = i_
                # XOR checks
                XOR_checks[iteration - n] = XORsum(self.currentIteration, possible[iteration - n].currentIteration)
                yes = True
                for a in XOR_Combos[iteration - n - 1]:
                    yes *= XOR_checks[positionMix[a][0]] ^ XOR_checks[positionMix[a][1]] == cs[a]
                if yes:
                    if isinstance(self.child, Item):
                        self.child.iterate(iteration + 1)
                    else:
                        self.child = sum([len(p.currentIteration) for p in possible[:n]])
                        [p.save() for p in possible]
        elif iteration == 0:
            # First BSL
            for i_ in self.iterable:
                self.currentIteration = i_
                # Length check
                if len(self.currentIteration) < possible[-1].child:
                    self.child.iterate(iteration + 1)
                else:
                    break
        else:
            for i_ in self.iterable:
                self.currentIteration = i_
                XOR_checks[0] = XORsum(self.currentIteration, possible[0].currentIteration)
                self.child.iterate(iteration + 1)

    def save(self):
        self.value = self.currentIteration


class Res:
    def __init__(self, time_taken, batch_size, l, BSL_length):
        self.time_taken = time_taken
        self.batch_size = batch_size
        self.l = l
        self.BSL_length = BSL_length

    def __repr__(self):
        return f'Time {self.time_taken}s for {self.batch_size} files of length {2 * self.l}. Length of BSL is {self.BSL_length}'


def XORsum(k, s):
    out = 0
    for i in s:
        out ^= k << i
    return out


def bitRep(e: int) -> str:
    return ''.join('{0:08b}'.format(e, 'b'))


def decode(encoded, k, s):
    return encoded ^ XORsum(k, s)


L = []
lengths = [3]
# n = 2, 3
k = 4
iterations = 1000

t_ = time()
for n in lengths:
    L.append([])
    s0 = []
    for i in range(1, k + 2):
        s0.extend([list(a) for a in list(combinations(range(k), i))])
    XOR_Combos = []
    for i in range(2, n + 1):
        XOR_Combos.append(list(combinations(range(i), 2)))

    included = []
    for i in range(len(XOR_Combos)):
        XOR_Combos[i] = [t for t in XOR_Combos[i] if t not in included]
        included.extend(XOR_Combos[i])

    del included

    positionMix = list(combinations(range(n), 2))
    XOR_Combos = [[positionMix.index(t) for t in p] for p in XOR_Combos]
    for _ in range(iterations):
        t0 = time()
        maxRand = 2 ** (2 * k) - 1
        fs = [randint(0, maxRand)]
        for _ in range(n - 1):
            temp = randint(0, maxRand)
            while temp in fs:
                temp = randint(0, maxRand)
            fs.append(temp)
        cs = [fs[i[0]] ^ fs[i[1]] for i in list(combinations(range(n), 2))]

        possible = []
        for i in range(n):
            possible.append(Item(s0))
        for i in range(n):
            possible.append(Item(range(2 ** (k + 1) - 1)))

        for i in range(2 * n - 1):
            possible[i].child = possible[i + 1]
        possible[-1].child = inf

        XOR_checks = [0] * n

        possible[0].iterate(0)

        if isinstance(possible[0].value, list):
            result = Res(time() - t0, n, k, sum([len(a.value) for a in possible[:n]]))
            print(result)
            L[-1].append(result)

timTot = 0
for i in L:
    timTot += sum([a.time_taken for a in i])
print(f'Time total (with processing)    - {time() - t_}\n'
      f'Time total (without processing) - {timTot}\n'
      f'Average time per iteration      - {1000 * timTot / iterations}')

print("    " + "S".ljust(19) + "Delta B_s".ljust(23) + "Delta B".ljust(23) + "R_s".ljust(23) + "R".ljust(23))
print("EM1")
for i in L:
    n = i[0].batch_size
    l = i[0].l
    delta = (l * (n - 2) - n) / log2(l)
    all = [p.BSL_length for p in i]
    S = [p < delta for p in all]
    allSuccess = [d for d, s in zip(all, S) if s]
    p = [f'{n} - {sum(S)}/{len(S)}={sum(S) / len(S)}' if len(S) != 0 else 'N/A',
         f'{sum([a * log2(l) + 2 * l + n - n * l for a in allSuccess]) / len(allSuccess)}' if len(allSuccess) != 0 else 'N/A',
         f'{sum([a * log2(l) + 2 * l + n - n * l for a in all]) / len(all)}',
         f'{round(sum([(a * log2(l) + 2 * l + n * (l + 1)) / (2 * n * l) for a in allSuccess]) / len(allSuccess), 5)}' if len(
             allSuccess) != 0 else 'N/A',
         f'{round(sum([(a * log2(l) + 2 * l + n * (l + 1)) / (2 * n * l) for a in all]) / len(all), 5)}']
    print(" | ".join([i.ljust(20) for i in p]))
print("EM2")
for i in L:
    n = i[0].batch_size
    l = i[0].l
    delta = (l * (n - 2)) / log2(l)
    all = [p.BSL_length for p in i]
    S = [p < delta for p in all]
    allSuccess = [d for d, s in zip(all, S) if s]
    p = [f'{n} - {sum(S)}/{len(S)}={sum(S) / len(S)}' if len(S) != 0 else 'N/A',
         f'{sum([a * log2(l) + 2 * l + n - 2 * n * l for a in allSuccess]) / len(allSuccess)}' if len(
             allSuccess) != 0 else 'N/A',
         f'{sum([a * log2(l) + 2 * l + n - 2 * n * l for a in all]) / len(all)}',
         f'{round(sum([(a * log2(l) + 2 * l) / (2 * n * l) for a in allSuccess]) / len(allSuccess), 5)}' if len(
             allSuccess) != 0 else 'N/A',
         f'{round(sum([(a * log2(l) + 2 * l) / (2 * n * l) for a in all]) / len(all), 5)}']
    print(" | ".join([i.ljust(20) for i in p]))
# print("EM2")
# for i in range(len(lengths)):
#     n = lengths[i]
#     delta = (2 * n * k) / (log2(k) - 1)
#     all = [sum([len(a) for a in p[:n]]) for p in L[i]]
#     S = [p < delta for p in all]
#     allSuccess = [d for d, s in zip(all, S) if s]
#     p = [f'{n} - {sum(S)}/{len(S)}={sum(S) / len(S)}' if len(S) != 0 else 'N/A',
#          f'{sum([a - n * k for a in allSuccess]) / len(allSuccess)}' if len(allSuccess) != 0 else 'N/A',
#          f'{sum([a - n * k for a in all]) / len(all)}',
#          f'{round(sum([(i * log2(k) + n * k) / (2 * n * k) for i in allSuccess]) / len(allSuccess), 5)}' if len(
#              allSuccess) != 0 else 'N/A',
#          f'{round(sum([(i * log2(k) + n * k) / (2 * n * k) for i in all]) / len(all), 5)}']
#     print(" | ".join([i.ljust(20) for i in p]))

# with open('test.txt', 'w') as f:
#     f.write("    " + "S".ljust(19) + "Delta B_s".ljust(23) + "Delta B".ljust(23) + "R_s".ljust(23) + "R".ljust(23) + "\n")
#     f.write('EM1' + "\n")
#     n = 3
#     for i in range(len(lengths)):
#         l = lengths[i]
#         delta = (n * l) / (l - 1)
#         S = [n < delta for n in L[i]]
#         allSuccess = [d for d, s in zip(L[i], S) if s]
#         p = [f'{2 * l} - {sum(S)}/{len(S)}={sum(S) / len(S)}' if len(S) != 0 else 'N/A',
#              f'{sum([i - n * l for i in allSuccess]) / len(allSuccess)}' if len(allSuccess) != 0 else 'N/A',
#              f'{sum([i - n * l for i in L[i]]) / len(L[i])}',
#              f'{round(sum([(i * log2(l) + n * l) / (2 * n * l) for i in allSuccess]) / len(allSuccess), 5)}' if len(
#                  allSuccess) != 0 else 'N/A',
#              f'{round(sum([(i * log2(l) + n * l) / (2 * n * l) for i in L[i]]) / len(L[i]), 5)}']
#         f.write(" | ".join([i.ljust(20) for i in p]) + "\n")
#     f.write('EM2' + "\n")
#     for i in range(len(L)):
#         l = lengths[i]
#         delta = (2 * n * l) / (l - 1)
#         S = [n < delta for n in L[i]]
#         allSuccess = [d for d, s in zip(L[i], S) if s]
#         p = [f'{2 * l} - {sum(S)}/{len(S)}={sum(S) / len(S)}' if len(S) != 0 else 'N/A',
#              f'{sum([i - n * l for i in allSuccess]) / len(allSuccess)}' if len(allSuccess) != 0 else 'N/A',
#              f'{sum([i - n * l for i in L[i]]) / len(L[i])}',
#              f'{round(sum([(i * log2(l)) / (2 * n * l) for i in allSuccess]) / len(allSuccess), 5)}' if len(
#                  allSuccess) != 0 else 'N/A',
#              f'{round(sum([(i * log2(l)) / (2 * n * l) for i in L[i]]) / len(L[i]), 5)}']
#         f.write(" | ".join([i.ljust(20) for i in p]) + "\n")
