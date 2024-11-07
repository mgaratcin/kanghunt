#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector>

// Structure Definitions
struct Steps128 {
    unsigned long long low;
    unsigned long long high;
} __attribute__((packed));

#pragma pack(push, 1)
struct Point {
    unsigned char high;        // 1 byte (bits 134–128)
    unsigned long long mid;    // 8 bytes (bits 127–64)
    unsigned long long low;    // 8 bytes (bits 63–0)
    Steps128 steps;            // 16 bytes
    unsigned char is_tame;     // 1 byte
} __attribute__((packed));
#pragma pack(pop)

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: validate_points <binary_file>" << std::endl;
        return 1;
    }

    const char* filename = argv[1];
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return 1;
    }

    size_t point_size = sizeof(Point);
    size_t index = 0;
    size_t non_compliant_count = 0;

    while (infile) {
        Point p;
        infile.read(reinterpret_cast<char*>(&p), point_size);
        if (infile.gcount() != point_size) {
            break; // End of file or read error
        }

        // Check if the lower 25 bits of 'low' are zero
        bool compliant = (p.low & ((1ULL << 25) - 1)) == 0;
        if (!compliant) {
            non_compliant_count++;
            std::cout << "Non-compliant Point at index " << index << ":\n";
            std::cout << "  high: 0x" << std::hex << static_cast<int>(p.high) << std::dec << "\n";
            std::cout << "  mid:  0x" << std::hex << p.mid << std::dec << "\n";
            std::cout << "  low:  0x" << std::hex << p.low << std::dec << "\n";
            std::cout << "  is_tame: " << static_cast<int>(p.is_tame) << "\n\n";
        }

        index++;
    }

    infile.close();

    if (non_compliant_count == 0) {
        std::cout << "All Points comply with the 25 trailing zero bits requirement." << std::endl;
    } else {
        std::cout << non_compliant_count << " non-compliant Points found." << std::endl;
    }

    return 0;
}
