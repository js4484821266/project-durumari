import numpy as np
from scipy.fft import dct, idct
import heapq
import time


# ============================================================
# 설정
# ============================================================

PATH = "Wildlife.wmv"

BLOCK_SIZE = 256

# 이전 실험에서 128 이하는 가망이 거의 없었으므로
# 256 근처를 자세히 조사
KEEP_LIST = [
    192,
    224,
    240,
    248,
    252,
    256,
]

# 한 번에 처리할 block 수
BATCH_BLOCKS = 2048


# ============================================================
# Huffman payload 길이 계산
# ============================================================

def huffman_bits_from_counts(counts):
    """
    frequency table만 가지고
    Huffman payload의 총 bit 수를 계산한다.

    실제 bitstream은 만들지 않는다.
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
# ============================================================

padded_size = (
    block_count * BLOCK_SIZE
)

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
# KEEP별 histogram
#
# quantized coefficient:
#     -127 ... +127
#     총 255 symbols
#
# residual:
#     -255 ... +255
#     총 511 symbols
# ============================================================

coef_counts = {
    keep: np.zeros(
        255,
        dtype=np.int64
    )
    for keep in KEEP_LIST
}

residual_counts = {
    keep: np.zeros(
        511,
        dtype=np.int64
    )
    for keep in KEEP_LIST
}


# approximation 품질 확인용
abs_error_sum = {
    keep: 0
    for keep in KEEP_LIST
}

sq_error_sum = {
    keep: 0
    for keep in KEEP_LIST
}


# ============================================================
# baseline: 원본 byte 자체 Huffman
# ============================================================

raw_counts = np.bincount(
    raw,
    minlength=256
)

raw_huffman_bits = (
    huffman_bits_from_counts(
        raw_counts
    )
)


# ============================================================
# 처리 시작
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

    x_u8 = blocks[
        batch_start:batch_end
    ]

    x = x_u8.astype(
        np.float64
    )


    # --------------------------------------------------------
    # 마지막 padding을 제외한 실제 byte 수
    # --------------------------------------------------------

    byte_start = (
        batch_start * BLOCK_SIZE
    )

    valid_count = min(
        x.size,
        original_size - byte_start
    )


    # ========================================================
    # block별 평균
    #
    # 실제 codec에서는 uint8 1 byte로 저장한다고 가정.
    # ========================================================

    mean = np.rint(
        np.mean(x, axis=1)
    )

    mean = np.clip(
        mean,
        0,
        255
    ).astype(np.int16)


    centered = (
        x - mean[:, None]
    )


    # ========================================================
    # DCT
    # ========================================================

    coef = dct(
        centered,
        type=2,
        norm="ortho",
        axis=1
    )


    # ========================================================
    # KEEP별 실험
    # ========================================================

    for keep in KEEP_LIST:

        # ----------------------------------------------------
        # 앞 KEEP개의 DCT coefficient
        # ----------------------------------------------------

        kept = coef[:, :keep]


        # ----------------------------------------------------
        # 각 block에서 가장 큰 coefficient를 ±127에 맞춤
        #
        # scale은 실제 저장할 때 float16 2 bytes.
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


        # 실제 저장되는 값은 float16이므로
        # encoder에서도 float16으로 반올림된 scale을 사용해야
        # decoder와 정확히 같은 결과가 나온다.
        scale16 = scale.astype(
            np.float16
        )

        scale_decoded = scale16.astype(
            np.float64
        )

        scale_decoded[
            scale_decoded == 0
        ] = 1.0


        # ====================================================
        # int8 coefficient quantization
        # ====================================================

        q = np.rint(
            kept
            / scale_decoded[:, None]
        )

        q = np.clip(
            q,
            -127,
            127
        ).astype(np.int8)


        # ----------------------------------------------------
        # coefficient Huffman histogram
        #
        # -127 ... +127
        #     ↓
        #   0 ... 254
        # ----------------------------------------------------

        q_symbols = (
            q.astype(np.int16) + 127
        ).ravel()

        coef_counts[keep] += np.bincount(
            q_symbols,
            minlength=255
        )


        # ====================================================
        # decoder가 reconstruct하는 DCT coefficients
        # ====================================================

        coef_restored = np.zeros(
            (len(x), BLOCK_SIZE),
            dtype=np.float64
        )

        coef_restored[:, :keep] = (
            q.astype(np.float64)
            * scale_decoded[:, None]
        )


        # ====================================================
        # IDCT
        # ====================================================

        predicted = idct(
            coef_restored,
            type=2,
            norm="ortho",
            axis=1
        )

        predicted += mean[:, None]


        # 실제 byte approximation
        predicted_u8 = np.clip(
            np.rint(predicted),
            0,
            255
        ).astype(np.uint8)


        # ====================================================
        # lossless residual
        #
        # original =
        #     predicted + residual
        #
        # 이 residual까지 저장하면 완전 무손실 복원이 가능.
        # ====================================================

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
        #
        # -255 ... +255
        #       ↓
        #    0 ... 510
        # ----------------------------------------------------

        residual_symbols = (
            residual_flat + 255
        ).astype(np.int64)

        residual_counts[keep] += np.bincount(
            residual_symbols,
            minlength=511
        )


        # ====================================================
        # approximation error
        # ====================================================

        r = residual_flat.astype(
            np.int64
        )

        abs_error_sum[keep] += int(
            np.sum(
                np.abs(r)
            )
        )

        sq_error_sum[keep] += int(
            np.sum(
                r * r
            )
        )


    # --------------------------------------------------------
    # 진행률
    # --------------------------------------------------------

    print(
        f"\rprocessed "
        f"{batch_end:,} / {block_count:,} blocks "
        f"({batch_end / block_count * 100:.1f}%)",
        end=""
    )


print()


# ============================================================
# 저장 비용 가정
# ============================================================

# 파일 전체 공통 header
#
# original size : uint64 = 8 bytes
# KEEP          : uint16 = 2 bytes
GLOBAL_HEADER_BYTES = 10


# canonical Huffman code length table을
# symbol 하나당 uint8 하나로 저장한다고 보수적으로 가정
#
# coefficient alphabet = 255
# residual alphabet    = 511
HUFFMAN_TABLE_BYTES = (
    255 + 511
)


# 각 block마다 반드시 저장해야 하는 metadata
#
# mean  : uint8   = 1 byte
# scale : float16 = 2 bytes
#
# coefficient 자체는 여기 포함하지 않음.
# 이번에는 Huffman payload로 따로 계산.
BLOCK_METADATA_BYTES = 3


metadata_bytes = (
    block_count
    * BLOCK_METADATA_BYTES
)


# ============================================================
# baseline 출력
# ============================================================

raw_huffman_bytes = (
    raw_huffman_bits + 7
) // 8

raw_total = (
    raw_huffman_bytes
    + 256          # raw Huffman code lengths
    + 8            # original size
)


print()
print("================ RAW HUFFMAN BASELINE ================")

print(
    "raw Huffman bits       :",
    raw_huffman_bits
)

print(
    "raw bits / byte        :",
    raw_huffman_bits / original_size
)

print(
    "raw Huffman bytes      :",
    raw_huffman_bytes
)

print(
    "raw estimated total    :",
    raw_total
)

print(
    "raw ratio              :",
    raw_total / original_size
)


# ============================================================
# KEEP별 최종 결과
# ============================================================

results = []

print()
print(
    "================================================================================================"
)

print(
    " KEEP | COEF bit/c | COEF MB | RES bit/B | RES MB | META MB | TOTAL MB | RATIO  | MAE"
)

print(
    "------------------------------------------------------------------------------------------------"
)


for keep in KEEP_LIST:

    # ========================================================
    # quantized DCT coefficients Huffman
    # ========================================================

    coef_bits = (
        huffman_bits_from_counts(
            coef_counts[keep]
        )
    )

    coef_bytes = (
        coef_bits + 7
    ) // 8


    # coefficient 하나당 평균 bit 수
    total_coef_count = (
        block_count * keep
    )

    coef_bits_per_symbol = (
        coef_bits / total_coef_count
    )


    # ========================================================
    # residual Huffman
    # ========================================================

    residual_bits = (
        huffman_bits_from_counts(
            residual_counts[keep]
        )
    )

    residual_bytes = (
        residual_bits + 7
    ) // 8


    residual_bits_per_byte = (
        residual_bits
        / original_size
    )


    # ========================================================
    # 총 저장량
    # ========================================================

    total_bytes = (
        GLOBAL_HEADER_BYTES
        + HUFFMAN_TABLE_BYTES
        + metadata_bytes
        + coef_bytes
        + residual_bytes
    )


    ratio = (
        total_bytes
        / original_size
    )


    mae = (
        abs_error_sum[keep]
        / original_size
    )

    rmse = np.sqrt(
        sq_error_sum[keep]
        / original_size
    )


    result = {
        "keep": keep,

        "coef_bits": coef_bits,
        "coef_bytes": coef_bytes,
        "coef_bits_per_symbol":
            coef_bits_per_symbol,

        "residual_bits":
            residual_bits,

        "residual_bytes":
            residual_bytes,

        "residual_bits_per_byte":
            residual_bits_per_byte,

        "metadata_bytes":
            metadata_bytes,

        "total_bytes":
            total_bytes,

        "ratio":
            ratio,

        "mae":
            mae,

        "rmse":
            rmse,
    }

    results.append(result)


    print(
        f"{keep:5d} | "
        f"{coef_bits_per_symbol:10.4f} | "
        f"{coef_bytes / 1e6:7.3f} | "
        f"{residual_bits_per_byte:9.4f} | "
        f"{residual_bytes / 1e6:6.3f} | "
        f"{metadata_bytes / 1e6:7.3f} | "
        f"{total_bytes / 1e6:8.3f} | "
        f"{ratio:6.4f} | "
        f"{mae:6.3f}"
    )


print(
    "================================================================================================"
)


# ============================================================
# 가장 작은 transform representation
# ============================================================

best = min(
    results,
    key=lambda r: r["total_bytes"]
)

elapsed = (
    time.perf_counter()
    - start_time
)


print()
print("================ BEST TRANSFORM RESULT ================")

print(
    "KEEP                       :",
    best["keep"]
)

print(
    "coefficient bits / symbol  :",
    best["coef_bits_per_symbol"]
)

print(
    "coefficient Huffman bytes  :",
    best["coef_bytes"]
)

print(
    "residual bits / byte       :",
    best["residual_bits_per_byte"]
)

print(
    "residual Huffman bytes     :",
    best["residual_bytes"]
)

print(
    "block metadata bytes       :",
    best["metadata_bytes"]
)

print(
    "final bytes                :",
    best["total_bytes"]
)

print(
    "original bytes             :",
    original_size
)

print(
    "final ratio                :",
    best["ratio"]
)

print(
    "approximation MAE          :",
    best["mae"]
)

print(
    "approximation RMSE         :",
    best["rmse"]
)

print(
    "elapsed                    :",
    elapsed,
    "seconds"
)


# ============================================================
# 직접 baseline과 비교
# ============================================================

print()
print("================ COMPARISON ================")

print(
    "Raw Huffman bytes          :",
    raw_total
)

print(
    "Best DCT+residual bytes    :",
    best["total_bytes"]
)

difference = (
    best["total_bytes"]
    - raw_total
)

print(
    "difference                 :",
    difference,
    "bytes"
)


if difference < 0:

    print(
        "SUCCESS: DCT model beat raw Huffman by",
        -difference,
        "bytes"
    )

else:

    print(
        "FAIL: raw Huffman was better by",
        difference,
        "bytes"
    )