from main import *
from itertools import *


class BSLHolder:
    def __init__(self, full_list, k):
        self.full_list = full_list
        self.list_length = len(self.full_list)
        self.split_list = [[self.full_list[0]]]
        self.previous_limit_expanded = []
        for i_ in self.full_list[1:]:
            appendable = []
            current = [[] for _ in range(k - 1)]
            for n_ in i_:
                current[sort(n_) - 1].append(n_)
            appendable.append(current)
            self.split_list.extend(appendable)

    def limit_expanded(self, threshold):
        for i_ in self.full_list[threshold]:
            self.previous_limit_expanded.append(i_)

    def limit_split(self, max_length, lower, higher):
        previous_limit_split = []
        for i_ in self.split_list[0]:
            previous_limit_split.extend(i_)
        for i_ in self.split_list[1:max_length]:
            for n_ in i_[max(0, higher - lower - 1):]:
                for p_ in n_:
                    if p_[0] > lower:
                        break
                    if p_[-1] >= higher:
                        previous_limit_split.append(p_)
        return previous_limit_split

    def reset(self):
        self.previous_limit_expanded = []


def sort_(e):
    return e[-1]


s0 = []
for i in range(1, 4 + 2):
    s0.append([list(a) for a in list(combinations(range(4), i))])

[i.sort(key=sort, reverse=True) for i in s0[1:]]
s0 = s0[:-1]

s0 = BSLHolder(s0, 4)

s0.limit_split(3, 0, 2)

iterable = []
threshold = 4
l = 4
n = 3
t = [[] for _ in range(l)]

t0 = time()

for i in range(l ** n):
    add = []
    for p in range(n):
        add.append(((i // (l ** p)) % l) + 1)
    t[max(add) - 1].append([*add, sum(add)])
[i.sort(key=sort_) for i in t]
print(f'{round((time() - t0) * 1000000, 3)}μs')
[print(i) for i in t]
