# -*- coding: utf-8 -*-

import sys
import struct

def toInt(data):
    return int(struct.unpack("<Q", data)[0])

result = int()

for chunk in iter(lambda: sys.stdin.buffer.read(8), b''):
    print(chunk)
    result += toInt(chunk)

print("Solutions: ");
print(result)



