import heapq
from collections import Counter
from itertools import count


class HuffmanNode:
    def __init__(self, symbol=None, left=None, right=None):
        self.symbol = symbol
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.left is None and self.right is None


def build_huffman_tree(data):
    """
    정수 배열 data로부터 Huffman tree를 만든다.
    """
    if not data:
        return None

    freq = Counter(data)

    # 빈도가 같을 때 노드끼리 직접 비교하지 않도록
    # 고유 번호를 같이 넣는다.
    unique = count()
    heap = []

    for symbol, f in freq.items():
        node = HuffmanNode(symbol=symbol)
        heapq.heappush(heap, (f, next(unique), node))

    # 심볼이 하나뿐인 경우
    if len(heap) == 1:
        return heap[0][2]

    while len(heap) > 1:
        f1, _, left = heapq.heappop(heap)
        f2, _, right = heapq.heappop(heap)

        parent = HuffmanNode(left=left, right=right)
        heapq.heappush(
            heap,
            (f1 + f2, next(unique), parent)
        )

    return heap[0][2]


def build_codebook(root):
    """
    Huffman tree로부터
    {정수: "0101..."} 형태의 코드북을 만든다.
    """
    if root is None:
        return {}

    codebook = {}

    def traverse(node, code):
        if node.is_leaf():
            # 데이터가 한 종류의 정수만 포함하는 경우
            codebook[node.symbol] = code if code else "0"
            return

        traverse(node.left, code + "0")
        traverse(node.right, code + "1")

    traverse(root, "")
    return codebook


def huffman_encode(data):
    """
    반환값:
        encoded_bits : "010101..." 문자열
        codebook     : {symbol: code}
        root         : Huffman tree
    """
    if not data:
        return "", {}, None

    root = build_huffman_tree(data)
    codebook = build_codebook(root)

    encoded_bits = "".join(codebook[x] for x in data)

    return encoded_bits, codebook, root


def huffman_decode(encoded_bits, root):
    """
    Huffman 비트열을 원래 정수 배열로 복호화한다.
    """
    if root is None:
        return []

    # 심볼 종류가 하나뿐인 경우
    if root.is_leaf():
        return [root.symbol] * len(encoded_bits)

    result = []
    node = root

    for bit in encoded_bits:
        if bit == "0":
            node = node.left
        elif bit == "1":
            node = node.right
        else:
            raise ValueError(f"잘못된 비트: {bit}")

        if node.is_leaf():
            result.append(node.symbol)
            node = root

    if node is not root:
        raise ValueError("비트열이 정상적인 Huffman code로 끝나지 않았습니다.")

    return result


# 사용 예시
data = [10, 10, 10, 20, 20, 30, 40, 40, 40, 40, -3, -3]

encoded, codebook, tree = huffman_encode(data)

print("원본:")
print(data)

print("\n빈도:")
print(Counter(data))

print("\n코드북:")
for symbol, code in codebook.items():
    print(symbol, "->", code)

print("\n부호화:")
print(encoded)

decoded = huffman_decode(encoded, tree)

print("\n복호화:")
print(decoded)

print("\n원본과 동일:", decoded == data)