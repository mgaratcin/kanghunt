// deploy_kangaroos.cpp

#include "deploy_kangaroos.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <sys/ioctl.h>
#include <unistd.h>
#include <cctype>
#include <atomic>
#include <chrono>
#include <random>
#include "secp256k1/SECP256k1.h"
#include "secp256k1/Point.h"
#include "secp256k1/Int.h"

static std::atomic<uint64_t> kangaroo_counter{0};

// Function prototype for the CUDA collision detection function
extern "C" void detect_collision(const char* dp1, const char* dp2, int length);

// Converts a hexadecimal string to a binary string representation
std::string hexToBinary(const std::string& hex) {
    std::string binary;
    for (char c : hex) {
        switch (std::tolower(c)) {
            case '0': binary += "0000"; break;
            case '1': binary += "0001"; break;
            case '2': binary += "0010"; break;
            case '3': binary += "0011"; break;
            case '4': binary += "0100"; break;
            case '5': binary += "0101"; break;
            case '6': binary += "0110"; break;
            case '7': binary += "0111"; break;
            case '8': binary += "1000"; break;
            case '9': binary += "1001"; break;
            case 'a': binary += "1010"; break;
            case 'b': binary += "1011"; break;
            case 'c': binary += "1100"; break;
            case 'd': binary += "1101"; break;
            case 'e': binary += "1110"; break;
            case 'f': binary += "1111"; break;
            default: break;
        }
    }
    return binary;
}

// Updates the Kangaroo Counter displayed on the terminal
void updateKangarooCounter(double power_of_two) {
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    int term_lines = w.ws_row;
    int term_cols = w.ws_col;

    std::ostringstream counter_message;
    counter_message << "[+] Local Dynamic Kangaroo Counter: 2^" << std::fixed << std::setprecision(5) << power_of_two;

    int start_col = term_cols - static_cast<int>(counter_message.str().length());
    if (start_col < 0) start_col = 0;

    std::cout << "\033[" << term_lines << ";" << start_col << "H";
    std::cout << "\033[K";
    std::cout << counter_message.str() << std::flush;
}

// Main function to deploy kangaroos for collision detection
void deploy_kangaroos(const std::vector<Int>& kangaroo_batch) {
    std::string dp1, dp2;
    static std::chrono::time_point<std::chrono::steady_clock> last_update_time = std::chrono::steady_clock::now();

    Secp256K1 secp;
    Point target_key;

    std::random_device rd;
    std::mt19937 gen(rd());
    uint64_t fixed_value = 150000000000000;

    for (const auto& base_key : kangaroo_batch) {
        Int current_key = base_key;

        // Make a non-const copy of base_key
        Int temp_base_key = base_key;

        // Print the base_key in binary if it has exactly 20 trailing zeros and a length of 136
        {
            std::string hex_str = temp_base_key.GetBase16();
            std::string binary_str = hexToBinary(hex_str);
            if (binary_str.length() == 136 && binary_str.substr(binary_str.size() - 20) == "00000000000000000000") {
                dp1 = binary_str;

                // Check if dp2 is not empty before calling detect_collision
                if (!dp2.empty()) {
                    detect_collision(dp1.c_str(), dp2.c_str(), dp1.length());
                }
            }
        }

        const int KANGAROO_JUMPS = 2048;
        for (int jump = 0; jump < KANGAROO_JUMPS; ++jump) {
            // Generate a 135-bit random value using multiple parts to ensure full precision
            Int jump_value;
            jump_value.SetInt64(fixed_value);               // Set the initial 64-bit part
            jump_value.ShiftL(64);                          // Shift left by 64 bits
            Int temp;
            temp.SetInt64(fixed_value);                     // Generate another random 64-bit value
            jump_value.Add(&temp);                          // Add to the jump_value
            jump_value.ShiftL(7);                           // Shift left by 7 more bits to target 135-bit size
            temp.SetInt64(fixed_value & ((1ULL << 7) - 1)); // Limit to the remaining 7 bits
            jump_value.Add(&temp);                          // Complete the 135-bit jump value

            // Update current_key based on the jump_value added to the base_key
            current_key.Add(&jump_value);

            Point current_pubkey = secp.ComputePublicKey(&current_key);

            if (current_pubkey.equals(target_key)) {
                std::cout << "\n[+] Target Key Found: " << current_key.GetBase16() << std::endl;
                return;
            }

            ++kangaroo_counter;

            // Print the current_key in binary if it has exactly 20 trailing zeros and a length of 136
            {
                std::string hex_str = current_key.GetBase16();
                std::string binary_str = hexToBinary(hex_str);
                if (binary_str.length() == 136 && binary_str.substr(binary_str.size() - 34) == "0000000000000000000000000000000000") {
                    dp2 = binary_str;

                    // Check if dp1 is not empty before calling detect_collision
                    if (!dp1.empty()) {
                        detect_collision(dp1.c_str(), dp2.c_str(), dp2.length());
                    }
                }
            }
        }

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_update_time).count() >= 2) {
            last_update_time = now;
            uint64_t current_count = kangaroo_counter.load();
            double power_of_two = log2(current_count);

            updateKangarooCounter(power_of_two);
        }
    }
}
