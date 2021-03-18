from modules import *


def string_list_to_int(list_slice):
    return int(''.join([str(i) for i in list_slice]), 2)


def parse(bits_to_parse):
    out = CompressedData()
    if bits_to_parse[0]:
        # Key is a nested compressed key
        del bits_to_parse[0]
        out.key = 0
        pass
    else:
        # Key is an actual value not compressed
        del bits_to_parse[0]
        out.key = string_list_to_int(bits_to_parse[:length + 1])
        del bits_to_parse[:length + 1]
        bsl_length = string_list_to_int(bits_to_parse[:2])
        for i in range(bsl_length):
            out.bsl.append(string_list_to_int(bits_to_parse[:log_l]))
            del bits_to_parse[:log_l]
    return out


class CompressedData:
    def __init__(self, key=0, bsl=None):
        self.key = key
        self.bsl = bsl if bsl else []

    def expand(self):
        if isinstance(self.key, CompressedData):
            self.key = self.key.expand()
        return XORsum(self.key, self.bsl)


with open('example.sqzr', 'r') as f:
    file = f.read()

bits = []
for char in file:
    char = ord(char)
    print(bitRep(char))
    for bit in range(8):
        bits.append((char >> (7 - bit)) % 2)

length = string_list_to_int(bits[:4])
log_l = int(log2(length))
