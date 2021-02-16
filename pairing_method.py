from itertools import combinations
from math import log2, inf
from random import *
from time import time


# So, this one is a good idea, and fast, but doesn't work.
# The keys always end up too long


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
    def __init__(self, length, file, index, child=None):
        self.value = 0
        self.length = length
        self.file_split = [file >> (2 * self.length - 3), file % 2]
        self.file = file
        self.current_iteration = 0
        self.child = child
        self.index = index

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
            min_s = (Dn & -Dn).bit_length() - 1
            max_s = Dn.bit_length() - self.length
            BitShiftList = possible[iteration - n].current_iteration
            if BitShiftList[0] <= min_s and BitShiftList[-1] >= max_s:
                yes = True
                first_xor = first ^ self.file_split[0]
                if first_xor:
                    yes = BitShiftList[-1] == self.length - 2
                if yes:
                    last_xor = last ^ self.file_split[1]
                    if last_xor:
                        yes = BitShiftList[0] == 0
                    if yes:
                        self.current_iteration = 0
                        BitShiftList = BitShiftList
                        if len(BitShiftList) > 1:
                            Dn >>= (sub := BitShiftList[0])
                            BitShiftList = [a - sub for a in BitShiftList]
                            for index in range(2 * self.length - 2):
                                self.current_iteration += (((Dn ^ self.XORsum_limited(self.current_iteration,
                                                                                      BitShiftList,
                                                                                      index)) >> index) % 2) << index
                                yes = XORsum(self.current_iteration, [a + sub for a in BitShiftList]) == Dn
                        else:
                            self.current_iteration = Dn >> BitShiftList[0]
                        if yes:
                            if isinstance(self.child, Key):
                                res = self.child.iterate(iteration + 1, F0C=F0C, first=first, last=last)
                                if (not res.type) and res.keep_on > 0:
                                    return res.less()
                            else:
                                self.child = sum([len(p.current_iteration) for p in possible[:n]])
                                [p.save() for p in possible]
                else:
                    return Iter_res(False, iteration - n)
        return Iter_res(True)

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

    def iterate(self, iteration, **kwargs):
        previous_length_sum = kwargs['previous_length_sum']
        for i_ in self.iterable:
            temp_previous_length_sum = previous_length_sum + len(i_)
            if temp_previous_length_sum < possible[-1].child:
                self.current_iteration = i_
                res = self.child.iterate(iteration + 1, previous_length_sum=temp_previous_length_sum)
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
iterations = 1

t_ = time()
for k in k_lengths:
    for n in n_lengths:
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
            for i in range(n):
                possible.append(Key(k + 1, fs[i], i + n))

            for i in range(2 * n - 1):
                possible[i].child = possible[i + 1]
            possible[-1].child = inf

            possible[0].iterate(0, previous_length_sum=0)

            # print(possible)

            if isinstance(possible[n].value, list):
                print(possible)
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
