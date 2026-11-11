import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

# ============================================================
# 전역 상수
# ============================================================

# 한 블록에 허용하는 최대 입력 길이.
# 0x100 == 256이다.
#
# 왜 256인가?
# - 입력 단위가 unsigned char이므로 원본 값 하나는 0~255이다.
# - 블록 내부 위치도 0~255라면 index를 unsigned char 1바이트로 저장할 수 있다.
# - 현재 실험은 256바이트 고정 블록을 기본 단위로 삼고 있다.
BLOCK_MAX = 0x100

# predictor residual 계산에 사용할 모듈러 값.
# 0x100 == 256이다.
#
# 예를 들어 원본 값이 3이고 predictor가 4라면 실제 차이는 -1이지만,
# unsigned char만 쓰기로 했으므로 -1 대신 255로 저장한다.
# 즉 residual = (3 - 4) mod 256 = 255이다.
BYTE_MOD = 0x100

# ============================================================
# 실제 테스트할 파일 경로
# ============================================================

# 여기에 직접 분석할 파일의 경로를 적는다.
# Windows 경로는 역슬래시(\)가 들어가므로 raw string인 r"..." 형태를 권장한다.
# 예:
# FILE_PATH = r"C:\\Users\\me\\Downloads\\bitcoin.pdf"
#
# Linux/macOS 예:
# FILE_PATH = "/home/me/Downloads/bitcoin.pdf"
#
# 빈 문자열로 두면 실제 파일 대신 아래의 내장 sanity test를 실행한다.
FILE_PATH = r""


# ============================================================
# 결과 자료형
# ============================================================

@dataclass(frozen=True)
class BasisResult:
    """
    predictor 하나를 적용한 뒤 얻은 결과를 한 묶음으로 보관한다.

    이 클래스를 따로 두는 이유는 모든 predictor가 똑같은 형태의 결과를
    반환하게 만들어 best_basis()에서 일관되게 비교하기 위해서다.

    Attributes
    ----------
    mode:
        어떤 predictor/표현 방식을 썼는지 나타내는 이름.
        현재는 사람이 보기 쉬운 문자열이지만, 실제 파일 포맷에서는
        0,1,2,... 같은 1바이트 mode id로 바꾸는 편이 낫다.

    params:
        decoder가 predictor를 재현하기 위해 필요한 파라미터 바이트열.
        예:
        - constant: 상수 predictor 값 1바이트
        - delta: 첫 원본 값 1바이트
        - delta2: 첫 두 원본 값 2바이트
        - periodic: 주기 T 1바이트 + 첫 T개 원본 seed

    residual:
        predictor가 틀린 위치만 저장한 sparse payload.
        현재 형식은 (index, residual_value) 쌍을 반복해서 저장한다.
        각 쌍은 index 1바이트 + value 1바이트 = 2바이트이다.

    cost:
        현재 임시 포맷에서 이 방식을 선택했을 때 필요한 예상 바이트 수.
        기본적으로 mode 1바이트 + params 길이 + residual 길이이다.

    nonzero:
        residual 중 0이 아닌 항의 개수.
        현재 sparse 포맷에서는 nonzero 하나당 2바이트를 쓰므로
        디버깅 및 압축성 확인용으로 유용하다.
    """

    mode: str
    params: bytes
    residual: bytes
    cost: int
    nonzero: int


# ============================================================
# 공통 유틸리티
# ============================================================

def _as_u8_array(barr: bytes) -> np.ndarray:
    """
    bytes 입력을 계산용 NumPy int64 배열로 바꾼다.

    절차
    ----
    1. np.frombuffer(..., dtype=np.uint8)로 원본 bytes를 0~255 배열로 본다.
    2. astype(np.int64)로 넓은 signed 정수형으로 복사한다.
    3. 길이가 256을 넘는지 검사한다.

    왜 바로 uint8로 계산하지 않는가?
    ----------------------------------
    uint8은 산술 중간값도 자동으로 0~255로 wrap된다.
    예를 들어 3 - 4가 -1이 아니라 곧바로 255가 된다.
    그러면 predictor 식을 디버깅하거나 중간값을 확인하기 불편하다.

    따라서 계산 중에는 int64를 쓰고,
    residual을 확정하는 순간 명시적으로 % 256을 적용한다.
    """

    # np.frombuffer는 bytes를 복사하지 않고 uint8 view를 만든다.
    # 즉 barr의 각 바이트가 곧 배열의 원소 하나가 된다.
    arr = np.frombuffer(barr, dtype=np.uint8)

    # predictor 산술에서 음수와 255 초과 값을 자연스럽게 다루기 위해
    # 충분히 넓은 int64로 변환한다.
    arr = arr.astype(np.int64)

    # 현재 포맷에서는 한 블록의 index를 1바이트로 저장하려 하므로
    # 최대 길이를 256으로 제한한다.
    if arr.size > BLOCK_MAX:
        raise ValueError("barr must contain at most 256 bytes")

    # 이후 predictor 함수들이 공통으로 사용할 계산용 배열을 반환한다.
    return arr


def _pack_sparse_residual(
    residual: np.ndarray,
    start: int = 0,
) -> tuple[bytes, int]:
    """
    residual 배열에서 0이 아닌 항만 골라 sparse payload로 직렬화한다.

    저장 형식
    ---------
    각 nonzero residual을 다음 2바이트로 저장한다.

        [index: 1 byte][value: 1 byte]

    예를 들어
        residual = [0, 0, 7, 0, 3]
    이면
        (2, 7), (4, 3)
    두 쌍만 저장한다.

    Parameters
    ----------
    residual:
        mod 256으로 정규화된 residual 배열.

    start:
        residual 검사 시작 index.
        predictor seed를 params에 따로 저장한 경우 그 앞부분은 건너뛴다.
        예:
        - delta는 x[0]을 seed로 저장하므로 start=1
        - delta2는 x[0], x[1]을 seed로 저장하므로 start=2
        - periodic T는 첫 T개를 seed로 저장하므로 start=T

    Returns
    -------
    payload:
        (index, value) 쌍들을 이어 붙인 bytes.

    nonzero:
        실제로 저장한 residual 개수.
    """

    # bytearray는 반복적으로 bytes를 붙일 때 bytes보다 효율적이다.
    out = bytearray()

    # 몇 개의 residual 항이 실제 파일에 들어갔는지 센다.
    nonzero = 0

    # seed로 이미 저장된 앞부분은 제외하고 검사한다.
    for i in range(start, residual.size):
        # NumPy scalar를 Python int로 바꾼다.
        r = int(residual[i])

        # 최종 파일에는 unsigned char 한 바이트만 저장하므로
        # 혹시 상위 비트가 있어도 0~255 범위로 강제한다.
        # residual이 올바르게 mod 256 되어 있다면 값은 원래 그대로다.
        r &= 0xFF

        # residual == 0은 predictor가 정확히 맞았다는 뜻이다.
        # 0은 저장하지 않는 것이 sparse encoding의 핵심이다.
        if r == 0:
            continue

        # struct.pack("<BB", i, r)
        # - '<' : little-endian 포맷 표기. B 한 바이트에는 endian 영향이 없지만
        #         전체 바이너리 포맷의 방향성을 일관되게 표시한다.
        # - 첫 B: 위치 index 0~255
        # - 둘째 B: residual 값 1~255
        out += struct.pack("<BB", i, r)

        # payload에 nonzero 하나를 추가했으므로 카운트를 증가시킨다.
        nonzero += 1

    # bytearray를 불변 bytes로 바꾸어 결과 객체에 넣기 쉽게 한다.
    payload = bytes(out)

    # 직렬화된 payload와 nonzero 개수를 함께 반환한다.
    return payload, nonzero


# ============================================================
# Predictor 1: constant / mode
# ============================================================

def constant_basis(barr: bytes) -> BasisResult:
    """
    블록 전체를 하나의 상수 c로 예측한다.

        predictor[n] = c
        residual[n] = x[n] - c mod 256

    여기서 c는 평균이 아니라 최빈값(mode)을 쓴다.

    왜 mode인가?
    ------------
    현재 residual 포맷은 nonzero 하나가 항상 2바이트이다.
    따라서 residual의 크기가 1인지 255인지는 저장비용에 거의 차이가 없다.
    중요한 것은 residual == 0인 위치를 최대화하는 것이다.

        residual[n] == 0  <=>  x[n] == c

    그러므로 원본에서 가장 자주 등장한 byte를 c로 고르면
    0 residual 개수가 최대가 된다.
    """

    # 공통 입력 변환과 길이 검사를 수행한다.
    arr = _as_u8_array(barr)

    # 빈 블록은 실제 데이터가 없으므로 predictor 값을 임의로 0으로 둔다.
    if arr.size == 0:
        # params에는 constant 값 0 한 바이트를 넣는다.
        params = b"\x00"

        # residual은 존재하지 않는다.
        payload = b""

        # 현재 포맷 가정:
        # mode id 1바이트 + constant parameter 1바이트 = 2바이트.
        cost = 2

        # nonzero residual은 0개이다.
        return BasisResult("constant", params, payload, cost, 0)

    # 0~255 각 byte 값이 몇 번 나왔는지 센다.
    # minlength=256을 주어 등장하지 않은 값도 빈도 0으로 포함한다.
    histogram = np.bincount(arr, minlength=BYTE_MOD)

    # 가장 많이 등장한 byte의 값을 predictor로 선택한다.
    # 동률이면 np.argmax 특성상 가장 작은 값이 선택된다.
    predictor = int(np.argmax(histogram))

    # 모든 원소에서 predictor를 뺀다.
    # 음수가 나와도 마지막에 % 256 하므로 0~255로 돌아온다.
    residual = (arr - predictor) % BYTE_MOD

    # constant predictor는 seed 위치가 따로 없으므로 start=0으로 전체 검사한다.
    payload, nonzero = _pack_sparse_residual(residual, start=0)

    # decoder가 같은 constant predictor를 만들 수 있도록 c를 1바이트로 저장한다.
    params = bytes([predictor])

    # 실제 예상 저장비용을 계산한다.
    # 1바이트 mode + 1바이트 constant + sparse residual payload.
    cost = 1 + len(params) + len(payload)

    # 모든 정보를 하나의 결과 객체로 묶어 반환한다.
    return BasisResult("constant", params, payload, cost, nonzero)


# ============================================================
# Predictor 2: delta / previous value
# ============================================================

def delta_basis(barr: bytes) -> BasisResult:
    """
    직전 값을 다음 값의 predictor로 사용한다.

        predictor[n] = x[n-1]
        residual[n] = x[n] - x[n-1] mod 256

    즉 전형적인 1차 delta coding이다.

    반복되는 데이터나 천천히 변하는 데이터에서 residual 0이 많이 생길 수 있다.
    """

    # 입력 bytes를 계산용 int64 배열로 바꾼다.
    arr = _as_u8_array(barr)

    # 입력이 비어 있으면 seed도 residual도 없다.
    if arr.size == 0:
        # mode id만 1바이트 있다고 가정한다.
        return BasisResult("delta", b"", b"", 1, 0)

    # 원본과 같은 길이의 residual 배열을 0으로 초기화한다.
    # residual[0]은 실제로 사용하지 않고 seed로 대체한다.
    residual = np.zeros_like(arr)

    # 원소가 2개 이상일 때만 x[n] - x[n-1]을 계산할 수 있다.
    if arr.size > 1:
        # arr[1:]  = x[1], x[2], ...
        # arr[:-1] = x[0], x[1], ...
        # 대응 원소끼리 빼면 1차 delta가 된다.
        diff = arr[1:] - arr[:-1]

        # unsigned char residual로 저장하기 위해 mod 256 한다.
        residual[1:] = diff % BYTE_MOD

    # x[0]은 predictor를 만들 이전 값이 없으므로 params에 seed로 저장한다.
    params = bytes([int(arr[0])])

    # residual[0]은 seed와 중복이므로 start=1부터 sparse encoding한다.
    payload, nonzero = _pack_sparse_residual(residual, start=1)

    # mode 1바이트 + seed 1바이트 + sparse payload.
    cost = 1 + len(params) + len(payload)

    # 결과를 반환한다.
    return BasisResult("delta", params, payload, cost, nonzero)


# ============================================================
# Predictor 3: delta2 / linear extrapolation
# ============================================================

def delta2_basis(barr: bytes) -> BasisResult:
    """
    앞의 두 값을 이용해 현재 값을 선형 외삽한다.

        predictor[n] = 2*x[n-1] - x[n-2]

    따라서 1차 차분이 일정한 수열에서는 residual이 0이 된다.

    예:
        10, 12, 14, 16, 18, ...

    에서는 이전 두 점의 기울기가 계속 +2이므로 정확히 예측된다.
    """

    # 입력을 계산용 배열로 변환한다.
    arr = _as_u8_array(barr)

    # 빈 입력이면 저장할 seed도 없다.
    if arr.size == 0:
        return BasisResult("delta2", b"", b"", 1, 0)

    # 입력이 1바이트뿐이면 2점 predictor를 만들 수 없다.
    if arr.size == 1:
        # 복원을 위해 그 한 바이트를 seed로 그대로 보관한다.
        params = bytes([int(arr[0])])

        # residual은 없다.
        payload = b""

        # mode 1바이트 + seed 1바이트.
        cost = 1 + len(params)

        return BasisResult("delta2", params, payload, cost, 0)

    # 원본 길이만큼 residual 배열을 0으로 준비한다.
    residual = np.zeros_like(arr)

    # n=2부터 각 위치의 predictor를 벡터 연산으로 계산한다.
    # arr[1:-1]은 x[n-1], arr[:-2]는 x[n-2]에 대응한다.
    pred = 2 * arr[1:-1] - arr[:-2]

    # predictor 자체도 unsigned char 공간에서 해석하므로 mod 256 한다.
    pred %= BYTE_MOD

    # 실제 값 x[n]과 예측값의 차이를 계산한다.
    diff = arr[2:] - pred

    # residual도 unsigned char 범위로 정규화한다.
    residual[2:] = diff % BYTE_MOD

    # 첫 두 값은 predictor 시작에 필요하므로 seed로 그대로 저장한다.
    params = bytes([int(arr[0]), int(arr[1])])

    # residual[0], residual[1]은 seed와 중복이므로 start=2부터 저장한다.
    payload, nonzero = _pack_sparse_residual(residual, start=2)

    # mode 1바이트 + seed 2바이트 + sparse payload.
    cost = 1 + len(params) + len(payload)

    # 결과를 반환한다.
    return BasisResult("delta2", params, payload, cost, nonzero)


# ============================================================
# Predictor 4: periodic
# ============================================================

def periodic_basis(
    barr: bytes,
    periods: Iterable[int] = (2, 3, 4, 5, 6, 7, 8, 16, 32),
) -> BasisResult:
    """
    이전 주기의 값을 predictor로 사용하는 방식.

        predictor[n] = x[n-T]
        residual[n] = x[n] - x[n-T] mod 256

    여러 후보 T를 시험하고 실제 예상 저장비용이 가장 작은 T를 선택한다.
    """

    # 입력을 계산용 배열로 변환한다.
    arr = _as_u8_array(barr)

    # 코드에서 반복해서 쓰므로 입력 길이를 n에 저장한다.
    n = arr.size

    # 빈 입력이면 주기를 선택할 이유가 없다.
    if n == 0:
        return BasisResult("periodic", b"", b"", 1, 0)

    # 아직 최적 후보를 찾지 못했다는 뜻으로 None에서 시작한다.
    best = None

    # 사용자가 제공한 각 후보 주기 T를 하나씩 시험한다.
    for T in periods:
        # T <= 0이면 주기로 의미가 없다.
        if T <= 0:
            continue

        # T >= n이면 x[n-T]를 참조할 수 있는 위치가 하나도 없거나 없다고 봐도 된다.
        if T >= n:
            continue

        # T 자체를 현재 포맷에서 1바이트로 저장하려 하므로 255를 넘기지 않는다.
        if T > 0xFF:
            continue

        # 현재 T에 대한 residual 배열을 0으로 초기화한다.
        residual = np.zeros_like(arr)

        # n >= T인 위치에서만 T칸 전 값을 predictor로 사용할 수 있다.
        # arr[T:]  = x[T], x[T+1], ...
        # arr[:-T] = x[0], x[1], ...
        diff = arr[T:] - arr[:-T]

        # unsigned char residual로 바꾼다.
        residual[T:] = diff % BYTE_MOD

        # 첫 T개 값은 과거 T칸 전 데이터가 없어서 예측할 수 없다.
        # 따라서 decoder 시작용 seed로 그대로 저장해야 한다.
        seed = bytes(arr[:T].tolist())

        # params의 첫 바이트에는 T를 저장하고 그 뒤에 seed T바이트를 붙인다.
        params = bytes([T]) + seed

        # 첫 T개 residual은 seed와 중복이므로 start=T부터 sparse encoding한다.
        payload, nonzero = _pack_sparse_residual(residual, start=T)

        # mode 1바이트 + T/seed params + sparse residual payload.
        cost = 1 + len(params) + len(payload)

        # 현재 T에 대한 완전한 결과 객체를 만든다.
        result = BasisResult(
            mode=f"periodic-{T}",
            params=params,
            residual=payload,
            cost=cost,
            nonzero=nonzero,
        )

        # 첫 유효 후보라면 무조건 best로 등록한다.
        if best is None:
            best = result
            continue

        # 이전 best보다 실제 예상 바이트 수가 작을 때만 교체한다.
        if result.cost < best.cost:
            best = result

    # 모든 T가 필터링되어 유효 후보가 하나도 없을 수 있다.
    if best is None:
        # 함수가 항상 BasisResult를 반환하도록 fallback을 만든다.
        # 여기서는 원본을 residual 자리에 그대로 넣는다.
        payload = bytes(arr.tolist())

        # mode 1바이트 + payload 길이로 비용을 잡는다.
        cost = 1 + len(payload)

        # 이 fallback에서는 모든 바이트를 저장했다고 보고 nonzero=n으로 둔다.
        return BasisResult("periodic", b"", payload, cost, n)

    # 가장 저렴했던 주기 후보를 반환한다.
    return best


# ============================================================
# Predictor 5: raw fallback
# ============================================================

def raw_basis(barr: bytes) -> BasisResult:
    """
    아무 predictor도 적용하지 않고 원본 블록을 그대로 저장한다.

    RAW는 실패가 아니라 반드시 필요한 fallback이다.
    무손실 압축에서는 모든 입력이 항상 짧아질 수 없기 때문에,
    predictor 메타데이터가 더 비싸면 그냥 원본을 저장하는 편이 낫다.
    """

    # 입력을 공통 검증하기 위해 배열로 변환한다.
    arr = _as_u8_array(barr)

    # RAW에서는 원본 byte 값을 그대로 payload로 만든다.
    payload = bytes(arr.tolist())

    # RAW는 predictor parameter가 없다.
    params = b""

    # mode 1바이트 + 원본 길이.
    cost = 1 + len(payload)

    # RAW의 nonzero는 predictor residual 의미가 없으므로
    # 단순히 payload byte 수를 통계값으로 넣는다.
    nonzero = len(payload)

    # 결과를 반환한다.
    return BasisResult("raw", params, payload, cost, nonzero)


# ============================================================
# 후보 선택기
# ============================================================

def best_basis(barr: bytes) -> BasisResult:
    """
    구현된 모든 후보를 실제 예상 직렬화 바이트 수로 비교한다.

    비교 대상
    ---------
    - raw
    - constant(mode)
    - delta
    - delta2
    - periodic

    핵심 원칙은 residual 제곱합이나 평균오차가 아니라
    실제로 파일에 몇 바이트를 쓰는지를 직접 비교하는 것이다.
    """

    # 각 후보 함수를 동일한 입력에 적용한다.
    raw_result = raw_basis(barr)
    constant_result = constant_basis(barr)
    delta_result = delta_basis(barr)
    delta2_result = delta2_basis(barr)
    periodic_result = periodic_basis(barr)

    # 결과들을 한 tuple로 묶는다.
    candidates = (
        raw_result,
        constant_result,
        delta_result,
        delta2_result,
        periodic_result,
    )

    # cost가 가장 작은 결과를 선택한다.
    # 동률이면 tuple에서 먼저 나온 후보가 선택된다.
    # 현재 RAW가 첫 번째라 동일 비용일 때 더 단순한 RAW를 우선한다.
    best = min(candidates, key=lambda result: result.cost)

    # 최종 선택 결과를 반환한다.
    return best


# ============================================================
# 실제 파일 분석용 함수
# ============================================================

def analyze_file(file_path: str | Path) -> None:
    """
    실제 파일을 읽어서 256바이트 블록 단위로 predictor 후보를 비교한다.

    이 함수의 목적은 아직 "압축 파일을 생성"하는 것이 아니다.
    현재 단계에서는 실제 파일을 넣었을 때 어떤 predictor가 얼마나 자주
    선택되는지, 그리고 우리가 계산한 예상 저장비용이 원본보다 작은지를
    빠르게 확인하는 벤치마크 도구 역할을 한다.

    처리 순서
    ---------
    1. 사용자가 지정한 경로를 Path 객체로 바꾼다.
    2. 파일을 binary mode로 연다.
    3. 정확히 256바이트씩 읽는다.
    4. 각 블록에 best_basis()를 적용한다.
    5. 선택된 mode와 예상 cost를 누적한다.
    6. 마지막에 파일 전체 통계를 출력한다.

    마지막 블록이 256바이트보다 짧아도 그대로 처리한다.
    _as_u8_array()가 최대 256바이트만 요구할 뿐 정확히 256바이트일 것을
    요구하지 않기 때문이다.
    """

    # 문자열 경로와 Path 경로를 모두 받을 수 있도록 Path로 통일한다.
    path = Path(file_path)

    # 경로가 실제 파일인지 먼저 확인한다.
    # 존재하지 않는 경로나 디렉터리를 실수로 넘겼을 때
    # 애매한 open 오류 대신 명확한 메시지를 주기 위함이다.
    if not path.is_file():
        raise FileNotFoundError(f"not a regular file: {path}")

    # 파일 전체 원본 바이트 수를 누적한다.
    original_bytes = 0

    # predictor를 선택했을 때의 예상 저장 바이트 수를 누적한다.
    # 현재 cost에는 각 블록의 mode 1바이트가 포함되어 있다.
    estimated_bytes = 0

    # 몇 개의 블록을 읽었는지 센다.
    block_count = 0

    # mode별 선택 횟수와 예상 저장비용을 집계하기 위한 dict다.
    # key는 "raw", "constant", "delta", "delta2", "periodic-4" 같은 mode 이름이다.
    mode_counts: dict[str, int] = {}
    mode_costs: dict[str, int] = {}

    # 실제 파일은 텍스트가 아니라 임의의 바이트열이므로 반드시 "rb"로 연다.
    with path.open("rb") as fp:
        while True:
            # 현재 설계의 기본 블록 크기인 256바이트만큼 읽는다.
            block = fp.read(BLOCK_MAX)

            # b""가 나오면 EOF에 도달한 것이다.
            if not block:
                break

            # 실제 읽힌 길이를 원본 크기에 더한다.
            # 마지막 블록은 256보다 작을 수 있다.
            original_bytes += len(block)

            # 이 블록에서 가장 저장비용이 싼 predictor를 찾는다.
            result = best_basis(block)

            # 선택된 predictor의 예상 저장비용을 누적한다.
            estimated_bytes += result.cost

            # 블록 수도 하나 증가시킨다.
            block_count += 1

            # mode별 선택 횟수를 센다.
            mode_counts[result.mode] = mode_counts.get(result.mode, 0) + 1

            # mode별로 실제 예상 몇 바이트를 소비했는지도 함께 센다.
            mode_costs[result.mode] = mode_costs.get(result.mode, 0) + result.cost

    # 빈 파일이면 비율 계산에서 0으로 나누게 되므로 따로 처리한다.
    if original_bytes == 0:
        print(f"File: {path}")
        print("Empty file")
        return

    # 예상 크기 / 원본 크기를 백분율로 계산한다.
    ratio = estimated_bytes / original_bytes * 100.0

    # 원본에서 예상 크기를 뺀 값이다.
    # 양수면 그만큼 줄었다는 뜻이고, 음수면 오히려 커졌다는 뜻이다.
    saved = original_bytes - estimated_bytes

    # --------------------------------------------------------
    # 파일 전체 요약 출력
    # --------------------------------------------------------
    print(f"File            : {path}")
    print(f"Original bytes  : {original_bytes:,}")
    print(f"Blocks          : {block_count:,}")
    print(f"Estimated bytes : {estimated_bytes:,}")
    print(f"Ratio           : {ratio:.3f}%")
    print(f"Saved bytes     : {saved:+,}")

    # --------------------------------------------------------
    # 어떤 predictor가 얼마나 자주 선택되었는지 출력
    # --------------------------------------------------------
    print("\nSelected modes:")

    # 선택 횟수가 많은 mode부터 보기 좋게 정렬한다.
    for mode, count in sorted(
        mode_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        # 해당 mode가 전체 블록 중 차지하는 비율을 계산한다.
        block_ratio = count / block_count * 100.0

        # 그 mode가 소비한 총 예상 바이트 수도 같이 보여준다.
        cost = mode_costs[mode]

        print(
            f"  {mode:12s} "
            f"blocks={count:6d} "
            f"({block_ratio:6.2f}%), "
            f"cost={cost:10,d} B"
        )


# ============================================================
# 내장 sanity test
# ============================================================

def run_sanity_tests() -> None:
    """
    파일 경로를 주지 않고 실행했을 때 사용하는 소형 자체 테스트다.

    실제 파일 분석 기능을 넣더라도 predictor 자체가 예상대로 동작하는지
    빠르게 확인할 수 있도록 기존 예제들을 별도 함수로 유지한다.
    """

    # predictor별 특성이 잘 드러나는 간단한 테스트 데이터 세트를 만든다.
    tests = {
        # 대부분 48이고 하나만 255인 블록.
        # constant(mode=48)가 가장 잘 먹혀야 한다.
        "single outlier": bytes([48] * 100 + [255] + [48] * 155),

        # 0,1,2,...,255 선형 ramp.
        # delta2 predictor가 거의 완벽하게 맞아야 한다.
        "ramp": bytes(range(256)),

        # 0,1,2,3 반복.
        # periodic T=4가 강해야 한다.
        "period-4": bytes([i % 4 for i in range(256)]),
    }

    # 테스트 이름과 데이터 블록을 하나씩 꺼낸다.
    for name, data in tests.items():
        # 테스트 이름을 눈에 잘 띄게 출력한다.
        print(f"\n[{name}]")

        # 각 후보 predictor를 개별적으로 실행해 비용을 비교한다.
        for fn in (
            raw_basis,
            constant_basis,
            delta_basis,
            delta2_basis,
            periodic_basis,
        ):
            # 현재 predictor를 테스트 데이터에 적용한다.
            result = fn(data)

            # mode, 예상 바이트 비용, nonzero residual 개수를 출력한다.
            print(
                f"{result.mode:12s} "
                f"cost={result.cost:4d} B, "
                f"nonzero={result.nonzero:3d}"
            )

        # 마지막으로 자동 선택기가 무엇을 골랐는지 출력한다.
        selected = best_basis(data)
        print("best:", selected)


# ============================================================
# 프로그램 진입점
# ============================================================

def main() -> None:
    """
    위쪽 FILE_PATH에 적어 둔 파일을 실제로 분석한다.

    FILE_PATH가 비어 있으면 파일을 열지 않고 내장 sanity test를 실행한다.
    따라서 테스트할 파일을 바꾸고 싶을 때는 명령행 인자를 쓸 필요 없이
    파일 위쪽의 FILE_PATH 한 줄만 수정하면 된다.
    """

    # strip()은 실수로 공백만 적어 둔 경우도 빈 경로로 취급하기 위해 사용한다.
    if FILE_PATH.strip():
        # 사용자가 코드에 직접 적어 둔 경로를 256바이트 블록 단위로 분석한다.
        analyze_file(FILE_PATH)
        return

    # FILE_PATH를 비워 두었을 때는 predictor 자체가 정상 동작하는지
    # 확인할 수 있도록 작은 예제들을 실행한다.
    run_sanity_tests()


# 이 파일을 직접 실행했을 때만 main()을 호출한다.
# 다른 Python 코드에서 import했을 때는 자동으로 파일 분석을 시작하지 않는다.
if __name__ == "__main__":
    main()
