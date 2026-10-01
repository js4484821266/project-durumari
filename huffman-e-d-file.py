import heapq
from collections import Counter

def make_codes(data):
    heap = [[count, [sym, ""]] for sym, count in Counter(data).items()]
    heapq.heapify(heap)

    if len(heap) == 1:
        sym = heap[0][1][0]
        return {sym: "0"}

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)

        for pair in lo[1:]:
            pair[1] = "0" + pair[1]

        for pair in hi[1:]:
            pair[1] = "1" + pair[1]

        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    return {sym: code for sym, code in heap[0][1:]}


def compress(data):
    codes = make_codes(data)
    bits = "".join(codes[b] for b in data)
    return codes, bits


def decompress(codes, bits):
    reverse = {code: sym for sym, code in codes.items()}

    out = bytearray()
    cur = ""

    for bit in bits:
        cur += bit

        if cur in reverse:
            out.append(reverse[cur])
            cur = ""

    return bytes(out)


# test
with open("huffman-e-d.py", "rb") as f:
    original = f.read()

codes, bits = compress(original)
restored = decompress(codes, bits)

print("original :", len(original), "bytes")
print("huffman  :", (len(bits) + 7) // 8, "bytes")
print("ratio    :", len(bits) / 8 / len(original))
print("roundtrip:", original == restored)