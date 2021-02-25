from itertools import combinations
from math import log2, inf
from random import *
from time import time


class IterRes:
    def __init__(self, passed: bool, BSL_length: int = 0):
        self.passed = passed
        self.BSL_length = BSL_length


class MasterPair:
    def __init__(self, master_BSL_iterable, child_BSL_iterable, Key_length, file, batch_size):
        self.BSL_iterable = master_BSL_iterable
        self.key_iterable = range(2 ** Key_length)
        self.key_length = Key_length
        self.file_length = 2 * (Key_length - 1)
        self.file_all = file
        self.file = file % 2 ** self.file_length
        self.batch_size = batch_size
        self.BSL_value = 0
        self.key_value = 0
        self.current_BSL = 0
        self.current_key = 0
        self.BSL_length_sum = inf
        self.rotate = 0
        self.children = []
        for i_ in range(1, self.batch_size):
            file = (self.file_all >> (self.file_length * i_)) % (2 ** self.file_length)
            self.children.append(BSL(child_BSL_iterable, Key(self.key_length, file), self))

    def iterate(self):
        for p_ in self.BSL_iterable:
            # for rotate in range(self.file_length):
            #     self.rotate = rotate
            #     self.bitRotate(self.file_all)
            #     self.rotateChildren()
            for i_ in p_:
                if len(i_) - (self.batch_size - 1) < self.BSL_length_sum:
                    self.current_BSL = i_
                    for n_ in self.key_iterable:
                        self.current_key = n_
                        F0C = self.file ^ XORsum(self.current_key, self.current_BSL)
                        cur_iter_len = len(i_)
                        while True:
                            for index, child in enumerate(self.children):
                                res = child.iterate(F0C, cur_iter_len, index)
                                if not res.passed:
                                    break
                                else:
                                    cur_iter_len += res.BSL_length
                            if not res.passed:
                                break
                            else:
                                self.BSL_length_sum = cur_iter_len
                                self.BSL_value = self.current_BSL
                                self.key_value = self.current_key
                                [p.save() for p in self.children]
                                if self.BSL_length_sum == self.batch_size:
                                    return

    def child_length(self, index):
        return self.batch_size - index - 2

    def bitRotate(self, e):
        first = e % 2
        self.file_all = (e >> 1) + (first << (self.batch_size * self.file_length))

    def rotateChildren(self):
        self.file = self.file_all % (2 ** self.file_length)
        for i_ in range(1, self.batch_size):
            self.children[i_ - 1].child.new_file((self.file_all >> (self.file_length * i_)) % (2 ** self.file_length))

    def __repr__(self, full: bool = False) -> str:
        out = f'{self.batch_size} files of length {self.file_length}. L(s) was {self.BSL_length_sum} and rotational ' \
              f'shift was {self.rotate}'
        if full:
            out += f'\nXORsum({bitRep(self.key_value, self.key_length)}, {self.BSL_value}) ^ ' \
                   f'{bitRep(self.file, self.file_length)} = ' \
                   f'{bitRep(XORsum(self.key_value, self.BSL_value) ^ self.file, self.file_length)}\n'
            out += '\n'.join([p.__repr__() for p in self.children])
            out += '\n'
        return out


class Key:
    def __init__(self, length, file):
        self.value = 0
        self.length = length
        self.file_split = [file >> (2 * self.length - 3), file % 2]
        self.file = file
        self.calculated = 0

    def iterate(self, Dn, BitShiftList) -> bool:
        if len(BitShiftList) == 1:
            self.calculated = Dn >> BitShiftList[0]
            return True
        else:
            self.calculated = 0
            Dn >>= (sub := BitShiftList[0])
            BitShiftList = [a - sub for a in BitShiftList]
            for index in range(self.length + BitShiftList[-1]):
                self.calculated += (((Dn ^ self.XORsum_limited(self.calculated,
                                                               BitShiftList,
                                                               index)) >> index) % 2) << index
            if XORsum(self.calculated, [a + sub for a in BitShiftList]) != Dn:
                return False
            return True

    def XORsum_limited(self, k, s, p) -> int:
        out = 0
        for bit_shift in s:
            if bit_shift > p:
                break
            elif bit_shift + self.length > p:
                out ^= k << bit_shift
        return out

    def new_file(self, e):
        self.file = e
        self.file_split = [self.file >> (2 * self.length - 3), self.file % 2]

    def save(self):
        self.value = self.calculated

    def __repr__(self):
        return f'{bitRep(self.value, self.length)}'


class BSL:
    def __init__(self, S0, child, parent):
        self.iterable = S0
        self.current_iteration = 0
        self.value = 0
        self.child = child
        self.parent = parent

    def iterate(self, F0C, previous_lengths, index) -> IterRes:
        Dn = F0C ^ self.child.file
        min_s = (Dn & -Dn).bit_length() - 1
        max_s = Dn.bit_length() - self.child.length
        for i_ in self.iterable:
            if self.parent.child_length(index) + len(i_) + previous_lengths < self.parent.BSL_length_sum:
                if i_[0] <= min_s and i_[-1] >= max_s:
                    self.current_iteration = i_
                    res = self.child.iterate(Dn, self.current_iteration)
                    if res:
                        return IterRes(True, len(self.current_iteration))
        return IterRes(False)

    def save(self) -> None:
        self.value = self.current_iteration
        self.child.save()

    def __repr__(self):
        return f'XORsum({bitRep(self.child.value, self.child.length)}, {self.value}) ^ ' \
               f'{bitRep(self.child.file, self.parent.file_length)}'


class Res:
    def __init__(self, time_taken, batch_size, l, BSL_length):
        self.time_taken = time_taken
        self.batch_size = batch_size
        self.l = l
        self.BSL_length = BSL_length

    def __repr__(self):
        return f'Time {round(self.time_taken, 5)}s for {self.batch_size} files of length {2 * self.l}. ' \
               f'Length of BSL is {self.BSL_length}'


def XORsum(k, s) -> int:
    out = 0
    for bit_shift in s:
        out ^= k << bit_shift
    return out


def bitRep(e: int, l: int = 8) -> str:
    return ''.join(('{0:0' + str(l) + 'b}').format(e, 'b'))


L = []
# Format: batch size(n), half file length(l)
lengths = [[3, 4]]
iterations = 1000

t_ = time()
if __name__ == '__main__':
    for n, k in lengths:
        L.append([])
        s0 = []
        s0M = []
        for i in range(1, k + 2):
            t = [list(a) for a in list(combinations(range(k), i))]
            s0.extend(t)
            s0M.append(t)

        for iteration_number in range(iterations):
            t0 = time()
            fs = randint(0, 2 ** (n * 2 * k) - 1)

            possible = MasterPair(s0M, s0, k + 1, fs, n)

            possible.iterate()

            if possible.BSL_length_sum != inf:
                result = Res(time() - t0, n, k, possible.BSL_length_sum)
                # print(f'{str(iteration_number).ljust(4)} - {result}')
                # print(possible.__repr__(full=False))
                print(str(iteration_number))
                L[-1].append(result)
            else:
                print('no')

    t_ = time() - t_

    print('\n\nAnalysis:')

    print('Time:')
    timTot = 0
    for i in L:
        timTot += sum([a.time_taken for a in i])
    print(f'Time total (with processing)    - {t_}\n'
          f'Time total (without processing) - {timTot}')
    for i in L:
        print(f'n={i[0].batch_size}, l={i[0].l} -> {sum([a.time_taken for a in i])}')

    print('\nL(s):')
    for i in L:
        mean = sum([a.BSL_length for a in i]) / len(i)
        min_ = min([a.BSL_length for a in i])
        max_ = max([a.BSL_length for a in i])
        print(
            f'n={i[0].batch_size}, l={i[0].l} -> {min_}(-{round(mean - min_, 5)}) - {mean} - {max_}(+{round(max_ - mean, 5)})')
    print('')
    print("      " + "S".ljust(17) + "| Delta B_s".ljust(25) + "| Delta B".ljust(25) + "| R_s".ljust(25) + "| R".ljust(
        25))
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
        print(" | ".join([i.ljust(22) for i in p]))
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
        print(" | ".join([i.ljust(22) for i in p]))
