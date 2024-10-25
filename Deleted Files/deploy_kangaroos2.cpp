//Copyright 2024 MGaratcin//

//All rights reserved.//

//This code is proprietary and confidential. Unauthorized copying, distribution,//
//modification, or any other use of this code, in whole or in part, is strictly//
//prohibited. The use of this code without explicit written permission from the//
//copyright holder is not permitted under any circumstances.//

#include "deploy_kangaroos.h"
#include <iostream>
#include "secp256k1/SECP256k1.h"
#include "secp256k1/Point.h"
#include "secp256k1/Int.h"
#include <random>
#include <atomic>
#include <cmath>
#include <iomanip>
#include <mutex>
#include <chrono>
#include <sys/ioctl.h>
#include <unistd.h>

#ifndef KANGAROO_BATCH_SIZE
#define KANGAROO_BATCH_SIZE 1024
#endif

#define TARGET_KEY "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16"

static std::atomic<uint64_t> kangaroo_counter{0};
static std::mutex output_mutex; // Mutex for thread-safe output

void updateKangarooCounter(double power_of_two) {
    // Lock mutex for thread-safe access
    std::lock_guard<std::mutex> lock(output_mutex);

    // Get terminal size
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    int term_lines = w.ws_row;
    int term_cols = w.ws_col;

    // Create the Kangaroo Counter message
    std::ostringstream counter_message;
    counter_message << "[+] Local Dynamic Kangaroo Counter: 2^" << std::fixed << std::setprecision(5) << power_of_two;

    // Calculate the starting column position to right-align the message
    int start_col = term_cols - static_cast<int>(counter_message.str().length());
    if (start_col < 0) start_col = 0; // Ensure it doesn't go out of bounds

    // Move the cursor to the bottom-right corner
    std::cout << "\033[" << term_lines << ";" << start_col << "H";
    // Clear the line from the cursor position
    std::cout << "\033[K";
    // Print the Kangaroo Counter
    std::cout << counter_message.str() << std::flush;
}

void deploy_kangaroos(const std::vector<Int>& kangaroo_batch) {
    static std::chrono::time_point<std::chrono::steady_clock> last_update_time = std::chrono::steady_clock::now();

    Secp256K1 secp;
    Point target_key;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint64_t> dis(1, 10000000000);

    for (const auto& base_key : kangaroo_batch) {
        Int current_key = base_key;

        const int KANGAROO_JUMPS = 2048;
        for (int jump = 0; jump < KANGAROO_JUMPS; ++jump) {
            Point current_pubkey = secp.ComputePublicKey(&current_key);

            if (current_pubkey.equals(target_key)) {
                std::lock_guard<std::mutex> lock(output_mutex);
                std::cout << "\n[+] Target Key Found: " << current_key.GetBase16() << std::endl;
                return;
            }

            Int jump_value;
            jump_value.SetInt64(dis(gen));
            current_key.Add(&jump_value);

            ++kangaroo_counter;

            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - last_update_time).count() >= 2) {
                last_update_time = now;
                uint64_t current_count = kangaroo_counter.load();
                double power_of_two = log2(current_count);

                updateKangarooCounter(power_of_two);
            }
        }
    }
}
