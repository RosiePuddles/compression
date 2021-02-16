from itertools import combinations
from math import log2, inf
from random import *
from time import time


class Key:
    def __init__(self, length, file, index, child=None):
        self.value = 0
        self.length = length
        self.file_split = [file >> (2 * self.length - 3), file % 2]
        self.file = file
        self.current_iteration = 0
        self.child = child
        self.index = 2 * index

    def iterate(self, iteration, **kwargs) -> int:
        previous_length_sum = kwargs['previous_length_sum']
        if iteration == 1:
            # First key
            for i_ in range(2 ** self.length):
                self.current_iteration = i_
                F0C = self.file ^ XORsum(self.current_iteration, possible[0].current_iteration)
                F0C = [F0C, F0C >> (2 * self.length - 3), F0C % 2]
                self.child.iterate(iteration + 1, previous_length_sum=previous_length_sum, F0C=F0C[0], first=F0C[1],
                                   last=F0C[2])
        else:
            # Not first key, so 'last' and 'first' exist
            F0C = kwargs['F0C']
            Dn = kwargs['Dn']
            first = kwargs['first']
            last = kwargs['last']
            passed_in = {'previous_length_sum': previous_length_sum, 'F0C': F0C, 'first': first, 'last': last}
            BitShiftList = possible[iteration - 1].current_iteration
            if len(BitShiftList) == 1:
                self.current_iteration = Dn >> BitShiftList[0]
                if isinstance(self.child, BSL):
                    res = self.child.iterate(iteration + 1, **passed_in)
                    return res
                else:
                    self.child = previous_length_sum
                    [p.save() for p in possible]
            else:
                self.current_iteration = 0
                Dn >>= (sub := BitShiftList[0])
                BitShiftList = [a - sub for a in BitShiftList]
                for index in range(self.length + BitShiftList[-1]):
                    self.current_iteration += (((Dn ^ self.XORsum_limited(self.current_iteration,
                                                                          BitShiftList,
                                                                          index)) >> index) % 2) << index
                if XORsum(self.current_iteration, [a + sub for a in BitShiftList]) != Dn:
                    return 1
                if isinstance(self.child, BSL):
                    res = self.child.iterate(iteration + 1, **passed_in)
                    return res
                else:
                    self.child = previous_length_sum
                    [p.save() for p in possible]
        return 1

    def XORsum_limited(self, k, s, p) -> int:
        out = 0
        for bit_shift in s:
            if bit_shift > p:
                break
            elif bit_shift + self.length > p:
                out ^= k << bit_shift
        return out

    def save(self):
        self.value = self.current_iteration

    def __repr__(self):
        return f'{bitRep(self.value, self.length)} {bitRep(self.file)} {bitRep(decode(self.file, self.value, possible[self.index].value))}'


class BSL:
    def __init__(self, S0, child=None):
        self.value = 0
        self.iterable = S0
        self.current_iteration = 0
        self.child = child

    def iterate(self, iteration, **kwargs) -> int:
        if iteration > 0:
            previous_length_sum = kwargs['previous_length_sum']
            F0C = kwargs['F0C']
            first = kwargs['first']
            last = kwargs['last']
            Dn = F0C ^ possible[iteration + 1].file
            min_s = (Dn & -Dn).bit_length() - 1
            max_s = Dn.bit_length() - possible[iteration + 1].length
            for i_ in self.iterable:
                if (temp_previous_length_sum := previous_length_sum + len(i_)) < possible[-1].child:
                    if i_[0] <= min_s and i_[-1] >= max_s:
                        self.current_iteration = i_
                        res = self.child.iterate(iteration + 1, previous_length_sum=temp_previous_length_sum, F0C=F0C,
                                                 first=first, last=last, Dn=Dn)
                        if res == 2:
                            return 2
            try:
                if res:
                    return 1
            except NameError:
                return 2
        else:
            for i_ in self.iterable:
                if (previous_length_sum := len(i_)) < possible[-1].child:
                    self.current_iteration = i_
                    self.child.iterate(1, previous_length_sum=previous_length_sum)
                else:
                    break
        return 0

    def save(self):
        self.value = self.current_iteration

    def __repr__(self):
        return f'{self.value}'


class Res:
    def __init__(self, time_taken, batch_size, l, BSL_length):
        self.time_taken = time_taken
        self.batch_size = batch_size
        self.l = l
        self.BSL_length = BSL_length

    def __repr__(self):
        return f'Time {self.time_taken}s for {self.batch_size} files of length {2 * self.l}. ' \
               f'Length of BSL is {self.BSL_length}'


def XORsum(k, s) -> int:
    out = 0
    for bit_shift in s:
        out ^= k << bit_shift
    return out


def bitRep(e: int, l: int = 8) -> str:
    return ''.join(('{0:0' + str(l) + 'b}').format(e, 'b'))


def decode(encoded, k, s):
    return encoded ^ XORsum(k, s)


L = []
n_lengths = [3]
k_lengths = [4]
iterations = 1000

t_ = time()
for n in n_lengths:
    for k in k_lengths:
        L.append([])
        s0 = []
        for i in range(1, k + 2):
            s0.extend([list(a) for a in list(combinations(range(k), i))])

        for iteration_number in range(iterations):
            t0 = time()
            maxRand = 2 ** (2 * k) - 1
            fs = [randint(0, maxRand)]
            for _ in range(n - 1):
                temp = randint(0, maxRand)
                while temp in fs:
                    temp = randint(0, maxRand)
                fs.append(temp)

            possible = []
            for i in range(n):
                possible.append(BSL(s0))
                possible.append(Key(k + 1, fs[i], i))

            for i in range(2 * n - 1):
                possible[i].child = possible[i + 1]
            possible[-1].child = inf

            possible[0].iterate(0)

            if isinstance(possible[0].value, list):
                # print(possible)
                result = Res(time() - t0, n, k, sum([len(a.value) for a in possible[::2]]))
                # print(f'{str(iteration_number).ljust(4)} - {result}')
                L[-1].append(result)
            else:
                print('no')

t_ = time() - t_
timTot = 0
for i in L:
    timTot += sum([a.time_taken for a in i])
print(f'Time total (with processing)    - {t_}\n'
      f'Time total (without processing) - {timTot}\n')

print(sum([a.BSL_length for a in L[0]]) / len(L[0]))

print("      " + "S".ljust(15) + "| Delta B_s".ljust(23) + "| Delta B".ljust(23) + "| R_s".ljust(23) + "| R".ljust(23))
print("EM1")
for i in L:
    n = i[0].batch_size
    l = i[0].l
    delta = (l * (n - 2) - n) / log2(l)
    all = [p.BSL_length for p in i]
    S = [p < delta for p in all]
    allSuccess = [d for d, s in zip(all, S) if s]
    p = [f'{n},{l} - {sum(S)}/{len(S)}={sum(S) / len(S)}' if len(S) != 0 else 'N/A',
         f'{round(sum([a * log2(l) + 2 * l + n - n * l for a in allSuccess]) / len(allSuccess), 5)}' if len(
             allSuccess) != 0 else 'N/A',
         f'{round(sum([a * log2(l) + 2 * l + n - n * l for a in all]) / len(all), 5)}',
         f'{round(sum([(a * log2(l) + 2 * l + n * (l + 1)) / (2 * n * l) for a in allSuccess]) / len(allSuccess), 5)}' if len(
             allSuccess) != 0 else 'N/A',
         f'{round(sum([(a * log2(l) + 2 * l + n * (l + 1)) / (2 * n * l) for a in all]) / len(all), 5)}']
    print(" | ".join([i.ljust(20) for i in p]))
print("EM2")
for i in L:
    n = i[0].batch_size
    l = i[0].l
    delta = (2 * l * (n - 2)) / log2(l)
    all = [p.BSL_length for p in i]
    S = [p < delta for p in all]
    allSuccess = [d for d, s in zip(all, S) if s]
    p = [f'{n},{l} - {sum(S)}/{len(S)}={sum(S) / len(S)}' if len(S) != 0 else 'N/A',
         f'{round(sum([a * log2(l) + 2 * l + n - 2 * n * l for a in allSuccess]) / len(allSuccess), 5)}' if len(
             allSuccess) != 0 else 'N/A',
         f'{round(sum([a * log2(l) + 2 * l + n - 2 * n * l for a in all]) / len(all), 5)}',
         f'{round(sum([(a * log2(l) + 2 * l) / (2 * n * l) for a in allSuccess]) / len(allSuccess), 5)}' if len(
             allSuccess) != 0 else 'N/A',
         f'{round(sum([(a * log2(l) + 2 * l) / (2 * n * l) for a in all]) / len(all), 5)}']
    print(" | ".join([i.ljust(20) for i in p]))
