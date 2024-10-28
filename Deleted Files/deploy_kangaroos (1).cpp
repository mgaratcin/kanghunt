// Copyright 2024 MGaratcin
// All rights reserved.
// This code is proprietary and confidential. Unauthorized copying, distribution,
// modification, or any other use of this code, in whole or in part, is strictly
// prohibited. The use of this code without explicit written permission from the
// copyright holder is not permitted under any circumstances.

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
extern "C" void process_keys_on_gpu(
    const uint8_t* dp1_keys_bytes, int dp1_count,
    const uint8_t* dp2_keys_bytes, int dp2_count,
    int key_size_bytes);

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

    // Move cursor to the bottom line and appropriate column
    std::cout << "\033[" << term_lines << ";" << start_col << "H";
    // Clear the line
    std::cout << "\033[K";
    // Output the counter message
    std::cout << counter_message.str() << std::flush;
}

// Function to serialize Int to bytes (big-endian)
void int_to_bytes(Int& value, uint8_t* bytes, int key_size_bytes) {
    const char* hex_cstr = value.GetBase16();
    std::string hex_str(hex_cstr);
    if (hex_str.length() < static_cast<size_t>(key_size_bytes * 2)) {
        hex_str = std::string(key_size_bytes * 2 - hex_str.length(), '0') + hex_str;
    }
    for (int j = 0; j < key_size_bytes; ++j) {
        bytes[j] = static_cast<uint8_t>(std::stoi(hex_str.substr(j * 2, 2), nullptr, 16));
    }
}

// Main function to deploy kangaroos for collision detection
void deploy_kangaroos(const std::vector<Int>& kangaroo_batch) {
    const int KEY_SIZE_BYTES = 17; // 136 bits
    std::vector<uint8_t> dp1_keys_bytes_batch;
    std::vector<uint8_t> dp2_keys_bytes_batch;

    static std::chrono::time_point<std::chrono::steady_clock> last_update_time = std::chrono::steady_clock::now();

    Secp256K1 secp;
    Point target_key; // Initialize your target_key as needed

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint64_t> dist(1, UINT64_MAX);

    const int KANGAROO_JUMPS = 250;
    const int BATCH_THRESHOLD = 50; // Adjust as needed

    for (auto base_key : kangaroo_batch) {
        Int current_key = base_key;

        // Serialize base_key to bytes and add to dp1 batch
        std::vector<uint8_t> dp1_key_bytes(KEY_SIZE_BYTES);
        int_to_bytes(base_key, dp1_key_bytes.data(), KEY_SIZE_BYTES);
        dp1_keys_bytes_batch.insert(dp1_keys_bytes_batch.end(), dp1_key_bytes.begin(), dp1_key_bytes.end());

        for (int jump = 0; jump < KANGAROO_JUMPS; ++jump) {
            // Kangaroo jump logic
            Int jump_value;
            jump_value.SetInt64(dist(gen));
            jump_value.ShiftL(64);
            Int temp;
            temp.SetInt64(dist(gen));
            jump_value.Add(&temp);
            jump_value.ShiftL(7);
            temp.SetInt64(dist(gen) & ((1ULL << 7) - 1));
            jump_value.Add(&temp);

            current_key.Add(&jump_value);

            Point current_pubkey = secp.ComputePublicKey(&current_key);
            if (current_pubkey.equals(target_key)) {
                std::cout << "\n[+] Target Key Found: " << current_key.GetBase16() << std::endl;
                return;
            }

            ++kangaroo_counter;

            // Serialize current_key to bytes and add to dp2 batch
            std::vector<uint8_t> dp2_key_bytes(KEY_SIZE_BYTES);
            int_to_bytes(current_key, dp2_key_bytes.data(), KEY_SIZE_BYTES);
            dp2_keys_bytes_batch.insert(dp2_keys_bytes_batch.end(), dp2_key_bytes.begin(), dp2_key_bytes.end());
        }

        // Batch processing when batch size reaches a threshold
        if ((dp1_keys_bytes_batch.size() / KEY_SIZE_BYTES) >= BATCH_THRESHOLD &&
            (dp2_keys_bytes_batch.size() / KEY_SIZE_BYTES) >= BATCH_THRESHOLD) {
            int dp1_count = dp1_keys_bytes_batch.size() / KEY_SIZE_BYTES;
            int dp2_count = dp2_keys_bytes_batch.size() / KEY_SIZE_BYTES;

            process_keys_on_gpu(
                dp1_keys_bytes_batch.data(), dp1_count,
                dp2_keys_bytes_batch.data(), dp2_count,
                KEY_SIZE_BYTES);

            // Clear batches
            dp1_keys_bytes_batch.clear();
            dp2_keys_bytes_batch.clear();
        }

        // Update Kangaroo Counter periodically
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_update_time).count() >= 2) {
            last_update_time = now;
            uint64_t current_count = kangaroo_counter.load();
            double power_of_two = log2(static_cast<double>(current_count));

            updateKangarooCounter(power_of_two);
        }
    }

    // Process any remaining keys
    if (!dp1_keys_bytes_batch.empty() && !dp2_keys_bytes_batch.empty()) {
        int dp1_count = dp1_keys_bytes_batch.size() / KEY_SIZE_BYTES;
        int dp2_count = dp2_keys_bytes_batch.size() / KEY_SIZE_BYTES;

        process_keys_on_gpu(
            dp1_keys_bytes_batch.data(), dp1_count,
            dp2_keys_bytes_batch.data(), dp2_count,
            KEY_SIZE_BYTES);

        dp1_keys_bytes_batch.clear();
        dp2_keys_bytes_batch.clear();
    }
}
