import argparse
from pathlib import Path

MAGIC = b"PYLZMA1\0"


def _matches(data, level):
    r = [None] * len(data)
    heads = {}
    prev = [-1] * len(data)
    depth = (4, 6, 8, 12, 18, 26, 38, 56, 80, 120)[level]
    for p in range(len(data) - 2):
        k = data[p : p + 3]
        c = heads.get(k, -1)
        heads[k] = p
        prev[p] = c
        bl = bd = n = 0
        while c >= 0 and p - c <= 65535 and n < depth:
            limit = min(258, len(data) - p)
            l = 0
            while l < limit and data[c + l] == data[p + l]:
                l += 1
            if l > bl:
                bl, bd = l, p - c
            c = prev[c]
            n += 1
        if bl >= 4:
            r[p] = (bl, bd)
    return r


def compress_data(data: bytes, level: int = 9) -> bytes:
    """LZMA의 dictionary/match 핵심을 직접 구현한 교육용 압축기."""
    data = bytes(data)
    if not 0 <= level <= 9:
        raise ValueError("level must be in 0..9")
    found = _matches(data, level)
    out = bytearray(MAGIC + len(data).to_bytes(8, "little"))
    p = 0
    while p < len(data):
        m = found[p]
        if m:
            l, d = m
            out.append(1)
            out.extend(d.to_bytes(2, "little"))
            out.append(l - 3)
            p += l
        else:
            out.extend((0, data[p]))
            p += 1
    return bytes(out)


def decompress_data(data: bytes) -> bytes:
    data = bytes(data)
    if len(data) < 16 or data[:8] != MAGIC:
        raise ValueError("invalid PYLZMA1 header")
    size = int.from_bytes(data[8:16], "little")
    out = bytearray()
    p = 16
    while len(out) < size:
        if p >= len(data):
            raise ValueError("truncated stream")
        tag = data[p]
        p += 1
        if tag == 0:
            if p >= len(data):
                raise ValueError("truncated literal")
            out.append(data[p])
            p += 1
        elif tag == 1:
            if p + 2 >= len(data):
                raise ValueError("truncated match")
            d = int.from_bytes(data[p : p + 2], "little")
            l = data[p + 2] + 3
            p += 3
            if not d or d > len(out) or len(out) + l > size:
                raise ValueError("invalid match")
            for _ in range(l):
                out.append(out[-d])
        else:
            raise ValueError("invalid token")
    if p != len(data):
        raise ValueError("trailing data")
    return bytes(out)


def compress_file(source, target, level=9):
    Path(target).write_bytes(compress_data(Path(source).read_bytes(), level))


def decompress_file(source, target):
    Path(target).write_bytes(decompress_data(Path(source).read_bytes()))


def main():
    p = argparse.ArgumentParser(description="pure Python LZMA-style compressor")
    s = p.add_subparsers(dest="mode", required=True)
    c = s.add_parser("compress")
    c.add_argument("source")
    c.add_argument("target")
    c.add_argument("-l", "--level", type=int, default=9)
    d = s.add_parser("decompress")
    d.add_argument("source")
    d.add_argument("target")
    a = p.parse_args()
    (
        compress_file(a.source, a.target, a.level)
        if a.mode == "compress"
        else decompress_file(a.source, a.target)
    )


if __name__ == "__main__":
    main()
