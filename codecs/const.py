import time
from collections import Counter


def encode(bys: bytes) -> list[int]:
    if not bys:
        return []

    mode, mode_count = Counter(bys).most_common(1)[0]

    exceptions = []

    for idx, val in enumerate(bys):
        if val != mode:
            exceptions.extend((idx, val))

    return [
        0x01,  # method = constant
        mode,  # 최빈값
        len(exceptions) // 2,  # 예외 개수
        *exceptions,
    ]


if __name__ == "__main__":
    test_str = b"4" * (0x100 - 2) + b"04"

    st = time.perf_counter()
    t = encode(test_str)
    tt = time.perf_counter() - st
    print(f"past {tt:.10}s; {int(1/tt)}/s")

    print(t)
    print(" ".join(hex(i) for i in t))
    print("encoded ratio :", len(t) / len(test_str))
    print("saving rate   :", 1 - len(t) / len(test_str))
