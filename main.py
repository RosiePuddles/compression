from itertools import combinations
from math import log2, inf
from random import *
from time import time


class Iter_res:
    def __init__(self, type, keep_on=0):
        """
        Iteration result class
        :param type: True  = No action required continue with iteration
                     False = Need to break a given number of times
        """
        self.type = type
        self.keep_on = keep_on

    def less(self):
        self.keep_on -= 1
        return self


class Key:
    def __init__(self, length, file, child=None):
        self.value = 0
        self.length = length
        self.file_split = [file >> (2 * self.length - 3), file % 2]
        self.file = file
        self.current_iteration = 0
        self.child = child

    def iterate(self, iteration, **kwargs) -> Iter_res:
        if iteration == n:
            # First key
            for i_ in range(2 ** self.length):
                self.current_iteration = i_
                F0C = self.file ^ XORsum(self.current_iteration, possible[0].current_iteration)
                F0C = [F0C, F0C >> (2 * self.length - 3), F0C % 2]
                res = self.child.iterate(iteration + 1, F0C=F0C[0], first=F0C[1], last=F0C[2])
                if (not res.type) and res.keep_on > 0:
                    return res.less()
        else:
            # Not first key, so 'last' and 'first' exist
            last = kwargs['last']
            first = kwargs['first']
            F0C = kwargs['F0C']
            Dn = self.file ^ F0C
            max_BSL = (Dn & -Dn).bit_length() - 1
            if possible[iteration - n].current_iteration[0] <= max_BSL:
                yes = True
                if first ^ self.file_split[0]:
                    yes = possible[iteration - n].current_iteration[-1] == self.length - 2
                if last ^ self.file_split[1]:
                    yes = possible[iteration - n].current_iteration[0] == 0
                if yes:
                    low = (first ^ self.file_split[0]) * (1 << (self.length - 1)) + (last ^ self.file_split[1])
                    if Dn.bit_length() - self.length > possible[iteration - n].current_iteration[-1]:
                        return Iter_res(False, n - 1)
                    else:
                        for i_ in range(low, 2 ** self.length, (last ^ self.file_split[1]) + 1):
                            self.current_iteration = i_
                            if self.file ^ XORsum(self.current_iteration,
                                                  possible[iteration - n].current_iteration) == F0C:
                                if iteration == 2 * n - 1:
                                    self.child = sum([len(p.current_iteration) for p in possible[:n]])
                                    [p.save() for p in possible]
                                else:
                                    res = self.child.iterate(iteration + 1, F0C=F0C, first=first, last=last)
                                    if (not res.type) and res.keep_on > 0:
                                        return res.less()
                else:
                    return Iter_res(False, iteration - n)
        return Iter_res(True)

    def save(self):
        self.value = self.current_iteration

    def __repr__(self):
        return f'{bitRep(self.value)}'


class BSL:
    def __init__(self, S0, child=None):
        self.value = 0
        self.iterable = S0
        self.current_iteration = 0
        self.child = child

    def iterate(self, iteration, **kwargs):
        previous_length_sum = kwargs['previous_length_sum']
        for i_ in self.iterable:
            previous_length_sum += len(i_)
            if previous_length_sum < possible[-1].child:
                self.current_iteration = i_
                res = self.child.iterate(iteration + 1, previous_length_sum=previous_length_sum)
                if (not res.type) and res.keep_on > 0:
                    return res.less()
            else:
                break
        return Iter_res(True)

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
        return f'Time {self.time_taken}s for {self.batch_size} files of length {2 * self.l}. Length of BSL is ' \
               f'{self.BSL_length}.\n   {self.iterations} were required of a maximum of {self.max_iterations} ' \
               f'({round(10000 * self.iterations / self.max_iterations, 5)}E-4)'


def XORsum(k, s) -> int:
    out = 0
    for bit_shift in s:
        out ^= k << bit_shift
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

    s0_length = len(s0)

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
        for i in range(n):
            possible.append(Key(k + 1, fs[i]))

        for i in range(2 * n - 1):
            possible[i].child = possible[i + 1]
        possible[-1].child = inf

        possible[0].iterate(0, previous_length_sum=0)

        if isinstance(possible[0].value, list):

            result = Res(time() - t0, n, k, sum([len(a.value) for a in possible[:n]]))
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
    p = [f'{n} - {sum(S)}/{len(S)}={sum(S) / len(S)}' if len(S) != 0 else 'N/A',
         f'{round(sum([a * log2(l) + 2 * l + n - 2 * n * l for a in allSuccess]) / len(allSuccess), 5)}' if len(
             allSuccess) != 0 else 'N/A',
         f'{round(sum([a * log2(l) + 2 * l + n - 2 * n * l for a in all]) / len(all), 5)}',
         f'{round(sum([(a * log2(l) + 2 * l) / (2 * n * l) for a in allSuccess]) / len(allSuccess), 5)}' if len(
             allSuccess) != 0 else 'N/A',
         f'{round(sum([(a * log2(l) + 2 * l) / (2 * n * l) for a in all]) / len(all), 5)}']
    print(" | ".join([i.ljust(20) for i in p]))