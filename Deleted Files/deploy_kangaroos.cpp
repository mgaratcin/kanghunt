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

#ifndef KANGAROO_BATCH_SIZE
#define KANGAROO_BATCH_SIZE 512
#endif

#define TARGET_KEY "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16"

static std::atomic<uint64_t> kangaroo_counter{0};
static std::mutex output_mutex; // Mutex for thread-safe output

void deploy_kangaroos(const std::vector<Int>& kangaroo_batch) {
    Secp256K1 secp; // Initialize the SECP256K1 context using the default constructor

    Point target_key;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint64_t> dis(1, 10000000000);

    for (const auto& base_key : kangaroo_batch) {
        Int current_key = base_key;

        // Perform a fixed number of jumps
        const int KANGAROO_JUMPS = 512;
        for (int jump = 0; jump < KANGAROO_JUMPS; ++jump) {
            // Compute the corresponding public key
            Point current_pubkey = secp.ComputePublicKey(&current_key);

            if (current_pubkey.equals(target_key)) {
                std::lock_guard<std::mutex> lock(output_mutex);
                std::cout << "\n[+] Target Key Found: " << current_key.GetBase16() << std::endl;
                return; // Stop if the target is found
            }

            Int jump_value;
            jump_value.SetInt64(dis(gen)); // Random value for the jump
            current_key.Add(&jump_value);

            ++kangaroo_counter;

            if (kangaroo_counter % 10000000 == 0) {
                uint64_t current_count = kangaroo_counter.load(); // Get the current value
                double power_of_two = log2(current_count);

                std::lock_guard<std::mutex> lock(output_mutex);
                // Move cursor to the start of the line, clear the line, and print the counter
                std::cout << "\r\033[K[+] Kangaroo Counter: 2^" << std::fixed << std::setprecision(5) << power_of_two << std::flush;
            }
        }
    }
}
