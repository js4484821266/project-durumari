#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace lzma_std {

constexpr uint32_t kMagic = 0x4C5A4D41u; // "LZMA"
constexpr size_t kMaxDistance = 4096;
constexpr size_t kMaxMatchLength = 255;
constexpr size_t kMinMatchLength = 4;

struct Match {
    uint16_t distance = 0;
    uint8_t length = 0;
    bool valid = false;
};

inline uint16_t read_u16_le(const std::vector<uint8_t>& data, size_t offset) {
    return static_cast<uint16_t>(data[offset]) |
           (static_cast<uint16_t>(data[offset + 1]) << 8);
}

inline void write_u16_le(std::vector<uint8_t>& data, uint16_t value) {
    data.push_back(static_cast<uint8_t>(value & 0xFFu));
    data.push_back(static_cast<uint8_t>((value >> 8) & 0xFFu));
}

inline std::vector<uint8_t> encode_lzma(const std::vector<uint8_t>& input) {
    std::vector<uint8_t> output;
    output.reserve(input.size() + 32);

    output.push_back(static_cast<uint8_t>(kMagic & 0xFFu));
    output.push_back(static_cast<uint8_t>((kMagic >> 8) & 0xFFu));
    output.push_back(static_cast<uint8_t>((kMagic >> 16) & 0xFFu));
    output.push_back(static_cast<uint8_t>((kMagic >> 24) & 0xFFu));

    const uint32_t size = static_cast<uint32_t>(input.size());
    output.push_back(static_cast<uint8_t>(size & 0xFFu));
    output.push_back(static_cast<uint8_t>((size >> 8) & 0xFFu));
    output.push_back(static_cast<uint8_t>((size >> 16) & 0xFFu));
    output.push_back(static_cast<uint8_t>((size >> 24) & 0xFFu));

    size_t pos = 0;
    while (pos < input.size()) {
        Match best{};
        const size_t maxDistance = std::min<size_t>(pos, kMaxDistance);

        for (size_t distance = 1; distance <= maxDistance; ++distance) {
            const size_t start = pos - distance;
            size_t length = 0;

            while (length < kMaxMatchLength &&
                   pos + length < input.size() &&
                   input[start + length] == input[pos + length]) {
                ++length;
            }

            if (length >= kMinMatchLength && length > best.length) {
                best.distance = static_cast<uint16_t>(distance);
                best.length = static_cast<uint8_t>(length);
                best.valid = true;
            }
        }

        if (best.valid) {
            output.push_back(0x01u);
            write_u16_le(output, best.distance);
            output.push_back(best.length);
            pos += best.length;
        } else {
            output.push_back(0x00u);
            output.push_back(input[pos]);
            ++pos;
        }
    }

    return output;
}

inline std::vector<uint8_t> decode_lzma(const std::vector<uint8_t>& input) {
    if (input.size() < 8) {
        return {};
    }

    const uint32_t magic = static_cast<uint32_t>(input[0]) |
                           (static_cast<uint32_t>(input[1]) << 8) |
                           (static_cast<uint32_t>(input[2]) << 16) |
                           (static_cast<uint32_t>(input[3]) << 24);

    if (magic != kMagic) {
        throw std::runtime_error("Invalid LZMA header");
    }

    const uint32_t original_size = static_cast<uint32_t>(input[4]) |
                                   (static_cast<uint32_t>(input[5]) << 8) |
                                   (static_cast<uint32_t>(input[6]) << 16) |
                                   (static_cast<uint32_t>(input[7]) << 24);

    std::vector<uint8_t> output;
    output.reserve(original_size);

    size_t pos = 8;
    while (output.size() < original_size && pos < input.size()) {
        const uint8_t tag = input[pos++];
        if (tag == 0x00u) {
            if (pos >= input.size()) {
                throw std::runtime_error("Unexpected end of stream while decoding literal");
            }
            output.push_back(input[pos++]);
        } else if (tag == 0x01u) {
            if (pos + 2 >= input.size()) {
                throw std::runtime_error("Unexpected end of stream while decoding match");
            }
            const uint16_t distance = read_u16_le(input, pos);
            pos += 2;
            if (pos >= input.size()) {
                throw std::runtime_error("Unexpected end of stream while decoding match length");
            }
            const uint8_t length = input[pos++];

            if (distance == 0 || distance > output.size()) {
                throw std::runtime_error("Invalid match distance");
            }

            for (uint8_t i = 0; i < length; ++i) {
                output.push_back(output[output.size() - distance]);
            }
        } else {
            throw std::runtime_error("Unknown token tag");
        }
    }

    if (output.size() != original_size) {
        throw std::runtime_error("Decoded size mismatch");
    }

    return output;
}

inline bool write_file(const std::string& path, const std::vector<uint8_t>& data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return true;
}

inline std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }

    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    in.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(static_cast<size_t>(size));
    if (size > 0) {
        in.read(reinterpret_cast<char*>(data.data()), size);
    }
    return data;
}

} // namespace lzma_std

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input-file> <output-file>\n";
        std::cerr << "The program stores a small, standard-library-only LZMA-style stream.\n";
        return 1;
    }

    try {
        const std::vector<uint8_t> input = lzma_std::read_file(argv[1]);
        if (input.empty()) {
            std::cerr << "Input file is empty or could not be opened.\n";
            return 1;
        }

        const std::vector<uint8_t> compressed = lzma_std::encode_lzma(input);
        if (!lzma_std::write_file(argv[2], compressed)) {
            std::cerr << "Failed to write compressed output.\n";
            return 1;
        }

        std::cout << "Compressed " << input.size() << " bytes into " << compressed.size() << " bytes.\n";

        const std::vector<uint8_t> decoded = lzma_std::decode_lzma(compressed);
        if (decoded != input) {
            std::cerr << "Round-trip verification failed.\n";
            return 1;
        }

        std::cout << "Round-trip verification passed.\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}
