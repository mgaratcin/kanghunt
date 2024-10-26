//Copyright 2024 MGaratcin//  
//All rights reserved.//
//This code is proprietary and confidential. Unauthorized copying, distribution,//
//modification, or any other use of this code, in whole or in part, is strictly//
//prohibited. The use of this code without explicit written permission from the//
//copyright holder is not permitted under any circumstances.//

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
#include <vector>
#include "secp256k1/SECP256k1.h"
#include "secp256k1/Point.h"
#include "secp256k1/Int.h"

#define TARGET_KEY "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16"

static std::atomic<uint64_t> kangaroo_counter{0};

// Function prototype for the CUDA collision detection function
extern "C" void detect_collision_batch(const uint8_t* dp1_batch, const uint8_t* dp2_batch, int length, int batch_size);

// Helper function to convert hexadecimal string to binary representation
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

// Function to update the Kangaroo Counter displayed on the terminal
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

// Helper function to convert binary string to byte vector
std::vector<uint8_t> binaryStringToBytes(const std::string& binary_str) {
    std::vector<uint8_t> bytes((binary_str.size() + 7) / 8, 0);
    for (size_t i = 0; i < binary_str.size(); ++i) {
        if (binary_str[i] == '1') {
            bytes[i / 8] |= (1 << (7 - (i % 8)));
        }
    }
    return bytes;
}

// Main function to deploy kangaroos for collision detection
void deploy_kangaroos(const std::vector<Int>& kangaroo_batch) {
    std::vector<std::vector<uint8_t>> dp1_batch;
    std::vector<std::vector<uint8_t>> dp2_batch;
    static std::chrono::time_point<std::chrono::steady_clock> last_update_time = std::chrono::steady_clock::now();

    Secp256K1 secp;
    Point target_key;

    std::random_device rd;
    std::mt19937 gen(rd());
    uint64_t fixed_value = 150000000000000;

    for (const auto& base_key : kangaroo_batch) {
        Int current_key = base_key;
        Int temp_base_key = base_key;

        // Collect dp1 data based on original constraints
        {
            std::string hex_str = temp_base_key.GetBase16();
            std::string binary_str = hexToBinary(hex_str);
            if (binary_str.length() == 136 && binary_str.substr(binary_str.size() - 20) == "00000000000000000000") {
                auto bytes = binaryStringToBytes(binary_str);
                dp1_batch.push_back(bytes);
            }
        }

        const int KANGAROO_JUMPS = 250;
        for (int jump = 0; jump < KANGAROO_JUMPS; ++jump) {
            Int jump_value;
            jump_value.SetInt64(fixed_value);
            jump_value.ShiftL(64);
            Int temp;
            temp.SetInt64(fixed_value);
            jump_value.Add(&temp);
            jump_value.ShiftL(7);
            temp.SetInt64(fixed_value & ((1ULL << 7) - 1));
            jump_value.Add(&temp);

            current_key.Add(&jump_value);

            Point current_pubkey = secp.ComputePublicKey(&current_key);
            if (current_pubkey.equals(target_key)) {
                std::cout << "\n[+] Target Key Found: " << current_key.GetBase16() << std::endl;
                return;
            }

            ++kangaroo_counter;

            // Collect dp2 data based on original constraints
            {
                std::string hex_str = current_key.GetBase16();
                std::string binary_str = hexToBinary(hex_str);
                if (binary_str.length() == 136 && binary_str.substr(binary_str.size() - 34) == "0000000000000000000000000000000000") {
                    auto bytes = binaryStringToBytes(binary_str);
                    dp2_batch.push_back(bytes);
                }
            }
        }

        // Batch processing when batch size reaches a threshold
        const int BATCH_THRESHOLD = 50; // Adjust as needed
        if (dp1_batch.size() >= BATCH_THRESHOLD && dp2_batch.size() >= BATCH_THRESHOLD) {
            // Ensure both batches have the same size
            size_t min_batch_size = std::min(dp1_batch.size(), dp2_batch.size());
            dp1_batch.resize(min_batch_size);
            dp2_batch.resize(min_batch_size);

            // Flatten the batches for GPU processing
            int length = dp1_batch[0].size();
            int batch_size = dp1_batch.size();

            std::vector<uint8_t> flat_dp1(batch_size * length);
            std::vector<uint8_t> flat_dp2(batch_size * length);

            for (int i = 0; i < batch_size; ++i) {
                std::copy(dp1_batch[i].begin(), dp1_batch[i].end(), flat_dp1.begin() + i * length);
                std::copy(dp2_batch[i].begin(), dp2_batch[i].end(), flat_dp2.begin() + i * length);
            }

            // Call the CUDA function
            detect_collision_batch(flat_dp1.data(), flat_dp2.data(), length, batch_size);

            // Clear batches
            dp1_batch.clear();
            dp2_batch.clear();
        }

        // Update Kangaroo Counter periodically
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_update_time).count() >= 2) {
            last_update_time = now;
            uint64_t current_count = kangaroo_counter.load();
            double power_of_two = log2(current_count);

            updateKangarooCounter(power_of_two);
        }
    }
}
