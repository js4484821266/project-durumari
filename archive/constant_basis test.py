import struct
import numpy as np
from archive.bases import *
if __name__ == "__main__":
    print(constant_basis(b"\x30" * 0x10 + b"\xff" + b"\x30" * 0xEF))
