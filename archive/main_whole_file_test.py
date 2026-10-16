import numpy as np
from sympy import isprime, nextprime, primerange

# primes
P = nextprime(1 << 8)

# Vandermonde
V = np.array([[pow(a, n, P) for n in range(P - 1)] for a in range(1, P)])

# Analysis
A = np.array([[-pow(a, -n, P) % P for n in range(P - 1)] for a in range(1, P)])

# Test File
PATH = "./Wildlife.wmv"

# Open Test File
f = open(PATH, "rb")
test_vector = np.frombuffer(f.read(P - 1), dtype=np.int8)
test_vector = np.array(test_vector, dtype=np.int64) % 0x100
print(test_vector)
f.close()

# Assertion
assert np.array_equal(V.T @ (A @ test_vector) % P, test_vector)
print("V.T @ (A @ test_vector) % P == test_vector")
assert np.array_equal(A @ (V.T @ test_vector) % P, test_vector)
print("A @ (V.T @ test_vector) % P == test_vector")


# ============================================================
# Whole-file compression experiment
# ============================================================
# 여기서부터는 파일의 첫 256바이트만 보는 것이 아니라 PATH의 파일 전체를 검사한다.
#
# 이번 실험에서 비교할 것은 두 가지다.
#
#   1) 원본 바이트 스트림을 그대로 Huffman 부호화했을 때의 payload 크기
#   2) 256바이트 블록마다 F_257 Fourier analysis(A)를 적용한 뒤,
#      나온 256개의 계수를 Huffman 부호화했을 때의 payload 크기
#
# 즉 현재 질문은 아주 단순하다.
#
#   "x 자체의 심볼 분포보다 A @ x mod 257의 심볼 분포가
#    Huffman coding에 더 유리한가?"
#
# 주의:
# - 아래의 크기는 Huffman 'payload'만 계산한다.
# - Huffman codebook/tree, 원본 파일 길이, 기타 헤더 비용은 아직 포함하지 않는다.
# - 따라서 이것은 완성된 파일 포맷의 압축률이 아니라 Fourier 변환 자체가
#   entropy coding에 도움이 되는지 확인하기 위한 실험이다.

import heapq
import os

BLOCK_SIZE = P - 1          # 257 - 1 = 256 bytes
BATCH_BLOCKS = 1024         # 한 번에 1024개 블록씩 계산해서 Python 반복 오버헤드를 줄임


def huffman_bits_from_counts(counts):
    """
    각 심볼의 출현 횟수(counts)만 가지고 이진 Huffman payload의 총 bit 수를 계산한다.

    실제 Huffman code를 문자열로 만들 필요는 없다.
    Huffman tree에서 두 최소 빈도를 합칠 때마다 그 합만큼 전체 code length가
    증가하므로, merge 비용의 총합이 곧 encoded payload의 총 bit 수다.

    counts[i] == 심볼 i의 출현 횟수
    """
    heap = [int(count) for count in counts if count > 0]

    # 심볼이 없거나 한 종류뿐이면, 원본 길이를 별도로 알고 있다는 전제에서
    # 이상적인 payload는 0 bit로 둘 수 있다.
    # 실제 파일 포맷에서는 codebook/header 비용이 따로 필요하다.
    if len(heap) <= 1:
        return 0

    heapq.heapify(heap)
    total_bits = 0

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        merged = left + right
        total_bits += merged
        heapq.heappush(heap, merged)

    return total_bits


# 원본은 0..255를 사용하지만 배열 길이를 P=257로 잡아 둔다.
# 이렇게 하면 Fourier 결과의 256 심볼까지 같은 함수/자료구조로 처리할 수 있다.
raw_counts = np.zeros(P, dtype=np.int64)
fourier_counts = np.zeros(P, dtype=np.int64)

file_size = os.path.getsize(PATH)
total_full_blocks = file_size // BLOCK_SIZE
processed_blocks = 0
batch_index = 0

with open(PATH, "rb") as f:
    while True:
        # 여러 256-byte block을 한 번에 읽는다.
        # Wildlife.wmv처럼 파일이 큰 경우 block 하나씩 A @ x를 호출하는 것보다
        # blocks @ A.T 형태의 행렬-행렬 계산이 훨씬 효율적이다.
        data = f.read(BLOCK_SIZE * BATCH_BLOCKS)

        if not data:
            break

        # ----------------------------------------------------
        # 1. 원본 stream의 histogram 누적
        # ----------------------------------------------------
        # 파일 전체에 대해 Huffman table 하나를 공유한다고 가정한다.
        raw_values = np.frombuffer(data, dtype=np.uint8)
        raw_counts[:256] += np.bincount(raw_values, minlength=256)

        # 이번 batch에서 Fourier transform할 수 있는 완전한 256-byte block 부분.
        full_size = (len(data) // BLOCK_SIZE) * BLOCK_SIZE

        if full_size > 0:
            blocks = np.frombuffer(
                data[:full_size], dtype=np.uint8
            ).astype(np.int64).reshape(-1, BLOCK_SIZE)

            # ------------------------------------------------
            # 2. 각 256-byte block에 finite-field Fourier analysis 적용
            # ------------------------------------------------
            # 한 block x에 대해서는 원래
            #
            #     coeff = A @ x % 257
            #
            # 이다.
            # 여기서는 block들이 행(row)으로 쌓여 있으므로 같은 계산을
            #
            #     blocks @ A.T
            #
            # 로 한 번에 수행한다.
            coeffs = (blocks @ A.T) % P

            # Fourier coefficient는 0..256까지 나올 수 있으므로 minlength=P.
            fourier_counts += np.bincount(
                coeffs.ravel(), minlength=P
            )

            processed_blocks += blocks.shape[0]

        # ----------------------------------------------------
        # 마지막 256바이트 미만 tail 처리
        # ----------------------------------------------------
        # A는 정확히 256차원 변환이므로 마지막 불완전 block에는 그대로 적용하지 않는다.
        # tail을 zero padding해서 Fourier 변환하면 padding 0도 압축 통계에 들어가
        # 비교가 왜곡될 수 있다.
        # 따라서 마지막 남는 바이트는 여기서는 그냥 raw symbol로 둔다.
        # decoder는 원본 파일 길이를 알면 tail 길이를 자동으로 알 수 있다.
        if full_size < len(data):
            tail = np.frombuffer(data[full_size:], dtype=np.uint8)
            fourier_counts[:256] += np.bincount(tail, minlength=256)

        batch_index += 1

        # 큰 파일에서 프로그램이 멈춘 것처럼 보이지 않도록 가끔 진행률 출력.
        if batch_index % 10 == 0 or processed_blocks == total_full_blocks:
            if total_full_blocks > 0:
                progress = 100.0 * processed_blocks / total_full_blocks
                print(
                    f"processed {processed_blocks:,} / "
                    f"{total_full_blocks:,} blocks ({progress:.1f}%)"
                )


# ============================================================
# Huffman payload size 계산
# ============================================================
raw_huffman_bits = huffman_bits_from_counts(raw_counts)
fourier_huffman_bits = huffman_bits_from_counts(fourier_counts)

raw_huffman_bytes = (raw_huffman_bits + 7) // 8
fourier_huffman_bytes = (fourier_huffman_bits + 7) // 8

original_bits = file_size * 8

print("\n================ Whole-file result ================")
print(f"file size                    : {file_size:,} bytes")
print(f"full 256-byte blocks         : {total_full_blocks:,}")
print(f"tail bytes                   : {file_size % BLOCK_SIZE}")

print("\n[Original bytes -> Huffman]")
print(f"Huffman payload              : {raw_huffman_bits:,} bits")
print(f"                              {raw_huffman_bytes:,} bytes")
print(f"ratio vs original file       : {raw_huffman_bits / original_bits:.6f}")

print("\n[Fourier coefficients -> Huffman]")
print(f"Huffman payload              : {fourier_huffman_bits:,} bits")
print(f"                              {fourier_huffman_bytes:,} bytes")
print(f"ratio vs original file       : {fourier_huffman_bits / original_bits:.6f}")

# 양수면 Fourier 변환 후 Huffman이 더 작다는 뜻.
# 음수면 원본을 바로 Huffman하는 쪽이 더 작다는 뜻.
difference_bits = raw_huffman_bits - fourier_huffman_bits

print("\n[Direct comparison]")
print(f"original Huffman - Fourier   : {difference_bits:,} bits")
print(f"                              {difference_bits / 8:.2f} bytes")

if difference_bits > 0:
    print("result                        : Fourier transform helped")
elif difference_bits < 0:
    print("result                        : Original bytes were better")
else:
    print("result                        : Tie")

# 분포 자체가 얼마나 달라졌는지 참고용으로 출력한다.
# 이것은 압축률을 직접 결정하는 값은 아니지만,
# Fourier transform이 심볼 종류를 더 퍼뜨렸는지/모았는지 확인하기 쉽다.
print("\n[Distribution]")
print(f"unique raw symbols           : {np.count_nonzero(raw_counts)}")
print(f"unique Fourier symbols       : {np.count_nonzero(fourier_counts)}")
