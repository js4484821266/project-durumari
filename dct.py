import numpy as np
from scipy.fft import dct, idct
import heapq
import time
import os

# ============================================================
# 설정
# ============================================================

PATH = "Wildlife.wmv"

BLOCK_SIZE = 256

# 몇 개의 저주파 DCT coefficient를 저장할지 비교
KEEP_LIST = [
    0,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    192,
    256,
]

# 메모리를 너무 많이 쓰지 않도록 여러 block을 묶어서 처리
BATCH_BLOCKS = 2048


# ============================================================
# Huffman payload bit 수 계산
# ============================================================

def huffman_bits_from_counts(counts):
    """
    symbol 빈도만 보고 Huffman payload가 정확히 몇 bit인지 계산.

    실제 Huffman bitstream을 만들지는 않는다.
    codebook 비용은 아래에서 따로 계산한다.
    """

    heap = [
        int(c)
        for c in counts
        if c > 0
    ]

    if len(heap) <= 1:
        return 0

    heapq.heapify(heap)

    total_bits = 0

    while len(heap) > 1:

        a = heapq.heappop(heap)
        b = heapq.heappop(heap)

        merged = a + b

        total_bits += merged

        heapq.heappush(
            heap,
            merged
        )

    return total_bits


# ============================================================
# 파일 읽기
# ============================================================

with open(PATH, "rb") as f:
    original_data = f.read()

original_size = len(original_data)

raw = np.frombuffer(
    original_data,
    dtype=np.uint8
)

block_count = (
    original_size + BLOCK_SIZE - 1
) // BLOCK_SIZE


# ============================================================
# 마지막 block padding
#
# 마지막 block이 256 byte보다 작으면 마지막 byte 값으로 padding.
#
# 실제 residual histogram과 오차 계산에는
# 원래 존재했던 byte만 포함한다.
# ============================================================

padded_size = block_count * BLOCK_SIZE

padded = np.empty(
    padded_size,
    dtype=np.uint8
)

padded[:original_size] = raw

if padded_size > original_size:

    if original_size > 0:
        padded[original_size:] = raw[-1]
    else:
        padded[:] = 0

blocks = padded.reshape(
    block_count,
    BLOCK_SIZE
)


# ============================================================
# KEEP마다 누적할 통계
#
# residual은
#
#     original - DCT approximation
#
# 이므로 -255 ~ +255.
#
# +255 해서 0 ~ 510의 511개 symbol로 만든다.
# ============================================================

counts = {
    keep: np.zeros(
        511,
        dtype=np.int64
    )
    for keep in KEEP_LIST
}

abs_error_sum = {
    keep: 0
    for keep in KEEP_LIST
}

sq_error_sum = {
    keep: 0
    for keep in KEEP_LIST
}


# ============================================================
# 실제 DCT model 저장 크기
#
# KEEP > 0일 때 block 하나:
#
# mean          1 byte
# scale         2 bytes (float16)
# coefficient   KEEP bytes (int8)
#
# 따라서:
#
#     3 + KEEP bytes/block
#
# KEEP=0은 DCT model 자체를 저장하지 않는다.
# 즉 순수 Huffman baseline.
# ============================================================

model_bytes = {}

for keep in KEEP_LIST:

    if keep == 0:
        model_bytes[keep] = 0

    else:
        model_bytes[keep] = (
            block_count
            * (3 + keep)
        )


# ============================================================
# 실험 시작
# ============================================================

start_time = time.perf_counter()

for batch_start in range(
    0,
    block_count,
    BATCH_BLOCKS
):

    batch_end = min(
        batch_start + BATCH_BLOCKS,
        block_count
    )

    # 현재 batch
    x_u8 = blocks[
        batch_start:batch_end
    ]

    x = x_u8.astype(
        np.float64
    )


    # --------------------------------------------------------
    # 실제 원본에서 이 batch가 차지하는 byte 개수
    #
    # 마지막 padding 부분은 제외해야 함.
    # --------------------------------------------------------

    byte_start = (
        batch_start
        * BLOCK_SIZE
    )

    valid_count = min(
        x.size,
        original_size - byte_start
    )


    # ========================================================
    # KEEP = 0
    #
    # DCT approximation 자체가 없음.
    #
    # residual = original
    #
    # 이것이 baseline:
    #
    #     원본 -> Huffman
    # ========================================================

    residual0 = (
        x_u8.astype(np.int16)
        .ravel()[:valid_count]
    )

    symbols0 = (
        residual0 + 255
    ).astype(np.int64)

    counts[0] += np.bincount(
        symbols0,
        minlength=511
    )


    # ========================================================
    # DCT를 사용하는 후보들
    # ========================================================

    # 각 256-byte block의 평균을 정수 byte 하나로 저장
    mean = np.rint(
        np.mean(x, axis=1)
    )

    mean = np.clip(
        mean,
        0,
        255
    ).astype(np.int16)


    # 평균 제거
    centered = (
        x
        - mean[:, None]
    )


    # DCT는 batch당 딱 한 번만 계산
    coef = dct(
        centered,
        type=2,
        norm="ortho",
        axis=1
    )


    # --------------------------------------------------------
    # KEEP별 reconstruction
    # --------------------------------------------------------

    for keep in KEEP_LIST:

        if keep == 0:
            continue


        # ----------------------------------------------------
        # 앞쪽 KEEP개의 coefficient만 사용
        # ----------------------------------------------------

        kept = coef[:, :keep]


        # ----------------------------------------------------
        # 각 block별로 coefficient를 int8로 양자화
        #
        # 가장 큰 절댓값을 ±127에 맞춤.
        # ----------------------------------------------------

        max_abs = np.max(
            np.abs(kept),
            axis=1
        )


        scale = np.where(
            max_abs == 0,
            1.0,
            max_abs / 127.0
        )


        # 실제 파일에는 float16으로 저장하므로
        # float16으로 바꿨다가 다시 사용한다.
        #
        # 즉 encoder/decoder에서 같은 scale을 사용.
        scale16 = scale.astype(
            np.float16
        )

        scale_decoded = scale16.astype(
            np.float64
        )


        # 혹시 float16 변환 후 0이 되는 경우 방지
        scale_decoded[
            scale_decoded == 0
        ] = 1.0


        quantized = np.rint(
            kept
            / scale_decoded[:, None]
        )

        quantized = np.clip(
            quantized,
            -127,
            127
        ).astype(np.int8)


        # ----------------------------------------------------
        # decoder가 실제로 얻는 DCT coefficients
        # ----------------------------------------------------

        coef_restored = np.zeros(
            (len(x), BLOCK_SIZE),
            dtype=np.float64
        )

        coef_restored[:, :keep] = (
            quantized.astype(np.float64)
            * scale_decoded[:, None]
        )


        # ----------------------------------------------------
        # IDCT
        # ----------------------------------------------------

        predicted = idct(
            coef_restored,
            type=2,
            norm="ortho",
            axis=1
        )


        # 저장해둔 평균 복원
        predicted += mean[:, None]


        # 실제 reconstructed byte
        predicted_u8 = np.clip(
            np.rint(predicted),
            0,
            255
        ).astype(np.uint8)


        # ----------------------------------------------------
        # lossless 복원을 위한 residual
        #
        # residual =
        #
        #     original - approximation
        #
        # 범위:
        #
        #     -255 ... +255
        # ----------------------------------------------------

        residual = (
            x_u8.astype(np.int16)
            - predicted_u8.astype(np.int16)
        )

        residual_flat = (
            residual.ravel()
            [:valid_count]
        )


        # ----------------------------------------------------
        # residual histogram
        # ----------------------------------------------------

        symbols = (
            residual_flat + 255
        ).astype(np.int64)

        counts[keep] += np.bincount(
            symbols,
            minlength=511
        )


        # ----------------------------------------------------
        # 오차 통계
        #
        # residual을 저장하기 때문에 최종 lossless decoder는
        # 정확히 복원되지만,
        #
        # DCT approximation 자체가 얼마나 좋은 predictor인지
        # 확인하기 위한 값.
        # ----------------------------------------------------

        r64 = residual_flat.astype(
            np.int64
        )

        abs_error_sum[keep] += int(
            np.sum(
                np.abs(r64)
            )
        )

        sq_error_sum[keep] += int(
            np.sum(
                r64 * r64
            )
        )


    # --------------------------------------------------------
    # 진행률
    # --------------------------------------------------------

    processed = batch_end

    print(
        f"\rprocessed "
        f"{processed:,} / {block_count:,} blocks "
        f"({processed / block_count * 100:.1f}%)",
        end=""
    )


print()


# ============================================================
# 실제 저장 포맷을 다음처럼 가정
#
# global header:
#
# original size : uint64   8 byte
# KEEP          : uint16   2 byte
#
# Huffman table:
#
# residual symbol은 -255~255 = 총 511개.
#
# 각 symbol의 canonical Huffman code length를
# uint8 하나씩 저장한다고 가정:
#
#     511 byte
#
# 즉 고정 header:
#
#     8 + 2 + 511 = 521 byte
#
# Huffman payload는 bit 단위이므로 마지막 byte는
# zero padding한다고 가정.
# ============================================================

HEADER_BYTES = 8 + 2 + 511


# ============================================================
# 결과 계산
# ============================================================

results = []

print()
print(
    "=========================================================================="
)
print(
    " KEEP | MODEL MB | RESID bits/B | RESID MB | TOTAL MB | RATIO  | MAE"
)
print(
    "--------------------------------------------------------------------------"
)


for keep in KEEP_LIST:

    huffman_bits = huffman_bits_from_counts(
        counts[keep]
    )

    huffman_bytes = (
        huffman_bits + 7
    ) // 8


    total_bytes = (
        HEADER_BYTES
        + model_bytes[keep]
        + huffman_bytes
    )


    ratio = (
        total_bytes
        / original_size
    )


    bits_per_original_byte = (
        huffman_bits
        / original_size
    )


    if keep == 0:

        mae = None
        rmse = None

        mae_text = "   -   "

    else:

        mae = (
            abs_error_sum[keep]
            / original_size
        )

        rmse = np.sqrt(
            sq_error_sum[keep]
            / original_size
        )

        mae_text = f"{mae:6.2f}"


    results.append({
        "keep": keep,
        "model_bytes": model_bytes[keep],
        "huffman_bits": huffman_bits,
        "huffman_bytes": huffman_bytes,
        "bits_per_byte": bits_per_original_byte,
        "total_bytes": total_bytes,
        "ratio": ratio,
        "mae": mae,
        "rmse": rmse,
    })


    print(
        f"{keep:5d} | "
        f"{model_bytes[keep] / 1e6:8.3f} | "
        f"{bits_per_original_byte:12.4f} | "
        f"{huffman_bytes / 1e6:8.3f} | "
        f"{total_bytes / 1e6:8.3f} | "
        f"{ratio:6.4f} | "
        f"{mae_text}"
    )


print(
    "=========================================================================="
)


# ============================================================
# 가장 작은 결과
# ============================================================

best = min(
    results,
    key=lambda x: x["total_bytes"]
)

elapsed = (
    time.perf_counter()
    - start_time
)


print()
print("================ BEST RESULT ================")

print(
    "KEEP                  :",
    best["keep"]
)

print(
    "DCT model bytes       :",
    best["model_bytes"]
)

print(
    "Residual Huffman bits :",
    best["huffman_bits"]
)

print(
    "Residual bits / byte  :",
    best["bits_per_byte"]
)

print(
    "Residual Huffman bytes:",
    best["huffman_bytes"]
)

print(
    "Final bytes           :",
    best["total_bytes"]
)

print(
    "Original bytes        :",
    original_size
)

print(
    "Final ratio           :",
    best["ratio"]
)

if best["mae"] is not None:

    print(
        "Approximation MAE     :",
        best["mae"]
    )

    print(
        "Approximation RMSE    :",
        best["rmse"]
    )

print(
    "Elapsed               :",
    elapsed,
    "seconds"
)

print()

if best["total_bytes"] < original_size:

    print(
        "SUCCESS:",
        original_size - best["total_bytes"],
        "bytes saved"
    )

else:

    print(
        "FAIL:",
        best["total_bytes"] - original_size,
        "bytes larger than original"
    )