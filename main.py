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
                if curLength < possible[-1].child:
                    self.child.iterate(iteration + 1)
        elif iteration > n:
            # Key not first
            for i_ in self.iterable:
                self.currentIteration = i_
                # XOR checks
                yes = True
                for a in XOR_Combos[iteration - n - 1]:
                    firstPair = [b.currentIteration for b in
                                 [possible[positionMix[a][0] + n], possible[positionMix[a][0]]]]
                    secondPair = [b.currentIteration for b in
                                  [possible[positionMix[a][1] + n], possible[positionMix[a][1]]]]
                    yes *= XORsum(*firstPair) ^ XORsum(*secondPair) == cs[a]
                if yes:
                    if isinstance(self.child, Item):
                        self.child.iterate(iteration + 1)
                    else:
                        self.child = sum([len(p.currentIteration) for p in possible[:n]])
                        [p.save() for p in possible]
        elif iteration == 0 or iteration == n:
            # First of it's type either key or BSL
            for i_ in self.iterable:
                self.currentIteration = i_
                self.child.iterate(iteration + 1)

    def save(self):
        self.value = self.currentIteration


def XORsum(k, s):
    out = 0
    for i in s:
        out ^= k << i
    return out


def bitRep(e: int) -> str:
    return ''.join('{0:08b}'.format(e, 'b'))


def decode(encoded, k, s):
    return encoded ^ XORsum(k, s)


def sort(e):
    return len(e[0][1]) + len(e[1][1]) + len(e[2][1])


L = []
lengths = [4]

for k in lengths:
    L.append([])
    s0 = []
    n = 2
    for i in range(1, k + 2):
        s0.extend([list(n) for n in list(combinations(range(k + 1), i))])
    for _ in range(10):
        t0 = time()
        fs = [randint(0, 2 ** (2 * k) - 1)]
        for _ in range(n - 1):
            temp = randint(0, 2 ** (2 * k) - 1)
            while temp in fs:
                temp = randint(0, 2 ** (2 * k) - 1)
            fs.append(temp)
        cs = [fs[i[0]] ^ fs[i[1]] for i in list(combinations(range(n), 2))]

        s0 = []
        for i in range(1, k + 2):
            s0.extend([list(n) for n in list(combinations(range(k + 1), i))])

        possible = []
        for i in range(n):
            possible.append(Item(s0))
        for i in range(n):
            possible.append(Item(range(2 ** k)))

        for i in range(2 * n - 1):
            possible[i].child = possible[i + 1]
        possible[-1].child = inf

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

        possible[0].iterate(0)

        print(time() - t0)
        f0 = fs[0] ^ XORsum(possible[n].value, possible[0].value)
        [print(f'{bitRep(fs[a])} = {bitRep(f0)} ^ XORsum({bitRep(possible[n + a].value)} {possible[a].value})') for a in range(n)]
        possible = [a.value for a in possible]
        print('')

        L[-1].append(possible)

    # print(L[-1])

with open('test.txt', 'w') as f:
    f.write("    " + "S".ljust(19) + "Delta B_s".ljust(23) + "Delta B".ljust(23) + "R_s".ljust(23) + "R".ljust(23) + "\n")
    f.write('EM1' + "\n")
    n = 3
    for i in range(len(lengths)):
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
