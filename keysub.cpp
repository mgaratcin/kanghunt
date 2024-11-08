#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <random>
#include <openssl/bn.h>
#include <secp256k1.h>
#include <secp256k1_ecdh.h>
#include <secp256k1_recovery.h>

// Function to convert a hex string to a byte array
std::vector<unsigned char> hex_to_bytes(const std::string& hex) {
    std::vector<unsigned char> bytes;
    bytes.reserve(hex.length() / 2);
    for (size_t i = 0; i < hex.length(); i += 2) {
        unsigned int byte;
        std::stringstream ss;
        ss << std::hex << hex.substr(i, 2);
        ss >> byte;
        bytes.push_back(static_cast<unsigned char>(byte));
    }
    return bytes;
}

// Function to convert a byte array to a hex string
std::string bytes_to_hex(const unsigned char* bytes, size_t len) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for(size_t i = 0; i < len; ++i) {
        ss << std::setw(2) << static_cast<int>(bytes[i]);
    }
    return ss.str();
}

// Function to generate a random scalar within 1-80 bits
bool generate_random_scalar(unsigned char* scalar) {
    // 80 bits = 10 bytes
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis64(0, 0xFFFFFFFFFFFFFFFFULL); // 64 bits
    std::uniform_int_distribution<uint16_t> dis16(0, 0xFFFF); // 16 bits

    uint64_t part1 = dis64(gen); // Lower 64 bits
    uint16_t part2 = dis16(gen); // Upper 16 bits to make 80 bits total

    // Initialize a 32-byte scalar with zeros
    std::vector<unsigned char> scalar_bytes(32, 0);

    // Set the lower 8 bytes (little-endian)
    for(int i = 0; i < 8; ++i) {
        scalar_bytes[i] = (part1 >> (8 * i)) & 0xFF;
    }

    // Set the next 2 bytes
    scalar_bytes[8] = part2 & 0xFF;
    scalar_bytes[9] = (part2 >> 8) & 0xFF;

    // Copy to scalar
    for(int i = 0; i < 32; ++i) {
        scalar[i] = scalar_bytes[i];
    }

    // Check if scalar is non-zero
    bool non_zero = false;
    for(int i = 0; i < 32; ++i) {
        if(scalar[i] != 0) {
            non_zero = true;
            break;
        }
    }
    return non_zero;
}

// Function to negate a scalar: neg_scalar = n - k mod n
bool negate_scalar(const unsigned char* scalar, unsigned char* neg_scalar) {
    // Define the curve order n for secp256k1
    const char* n_hex = "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141";
    BIGNUM* n = BN_new();
    BIGNUM* k = BN_new();
    BIGNUM* neg_k = BN_new();
    BN_CTX* ctx = BN_CTX_new();

    if(!n || !k || !neg_k || !ctx) {
        std::cerr << "Failed to allocate BIGNUMs.\n";
        if(n) BN_free(n);
        if(k) BN_free(k);
        if(neg_k) BN_free(neg_k);
        if(ctx) BN_CTX_free(ctx);
        return false;
    }

    // Convert n_hex to BIGNUM
    if(!BN_hex2bn(&n, n_hex)) {
        std::cerr << "Failed to convert curve order to BIGNUM.\n";
        BN_free(n); BN_free(k); BN_free(neg_k); BN_CTX_free(ctx);
        return false;
    }

    // Convert scalar to BIGNUM
    if(!BN_bin2bn(scalar, 32, k)) {
        std::cerr << "Failed to convert scalar to BIGNUM.\n";
        BN_free(n); BN_free(k); BN_free(neg_k); BN_CTX_free(ctx);
        return false;
    }

    // Compute neg_k = n - k
    if(!BN_sub(neg_k, n, k)) {
        std::cerr << "Failed to compute negated scalar.\n";
        BN_free(n); BN_free(k); BN_free(neg_k); BN_CTX_free(ctx);
        return false;
    }

    // Convert neg_k back to bytes
    int bn_len = BN_num_bytes(neg_k);
    if(bn_len > 32) {
        std::cerr << "Negated scalar exceeds 32 bytes.\n";
        BN_free(n); BN_free(k); BN_free(neg_k); BN_CTX_free(ctx);
        return false;
    }

    // Initialize neg_scalar with zeros
    for(int i = 0; i < 32; ++i) {
        neg_scalar[i] = 0;
    }

    // Allocate buffer for neg_k
    unsigned char* neg_k_bin = new unsigned char[bn_len];
    if(!neg_k_bin) {
        std::cerr << "Failed to allocate memory for neg_k_bin.\n";
        BN_free(n); BN_free(k); BN_free(neg_k); BN_CTX_free(ctx);
        return false;
    }

    int neg_k_bytes = BN_bn2bin(neg_k, neg_k_bin);
    if(neg_k_bytes > 32) {
        std::cerr << "Negated scalar binary length exceeds 32 bytes.\n";
        delete[] neg_k_bin;
        BN_free(n); BN_free(k); BN_free(neg_k); BN_CTX_free(ctx);
        return false;
    }

    // Copy neg_k_bin to the end of neg_scalar (big-endian)
    for(int i = 0; i < neg_k_bytes; ++i) {
        neg_scalar[32 - neg_k_bytes + i] = neg_k_bin[i];
    }

    delete[] neg_k_bin;
    BN_free(n); BN_free(k); BN_free(neg_k); BN_CTX_free(ctx);
    return true;
}

int main(int argc, char* argv[]) {
    if(argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <target_pubkey_hex> <number_of_keys> <output_file>\n";
        return 1;
    }

    std::string target_pubkey_hex = argv[1];
    int number_of_keys = std::stoi(argv[2]);
    std::string output_file = argv[3];

    // Initialize secp256k1 context
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    if(!ctx) {
        std::cerr << "Failed to create secp256k1 context.\n";
        return 1;
    }

    // Convert target public key from hex to bytes
    std::vector<unsigned char> target_pubkey_bytes = hex_to_bytes(target_pubkey_hex);
    if(target_pubkey_bytes.size() != 33) {
        std::cerr << "Invalid public key length. Must be 33 bytes (compressed).\n";
        secp256k1_context_destroy(ctx);
        return 1;
    }

    // Parse the target public key
    secp256k1_pubkey target_pubkey;
    if(!secp256k1_ec_pubkey_parse(ctx, &target_pubkey, target_pubkey_bytes.data(), target_pubkey_bytes.size())) {
        std::cerr << "Failed to parse the target public key. Ensure it is valid.\n";
        secp256k1_context_destroy(ctx);
        return 1;
    }

    // Open the output file
    std::ofstream outfile(output_file);
    if(!outfile.is_open()) {
        std::cerr << "Failed to open the output file: " << output_file << "\n";
        secp256k1_context_destroy(ctx);
        return 1;
    }

    std::cout << "Generating " << number_of_keys << " subtracted public keys...\n";

    for(int i = 0; i < number_of_keys; ++i) {
        unsigned char scalar[32];
        // Generate a valid random scalar within 1-80 bits
        bool valid = false;
        while(!valid) {
            valid = generate_random_scalar(scalar);
        }

        // Negate the scalar: neg_scalar = n - k
        unsigned char neg_scalar[32];
        if(!negate_scalar(scalar, neg_scalar)) {
            std::cerr << "Failed to negate scalar.\n";
            continue;
        }

        // Initialize new_pubkey as a copy of target_pubkey
        secp256k1_pubkey new_pubkey = target_pubkey;

        // Tweak the public key: new_pubkey = P + (-k)G
        if(!secp256k1_ec_pubkey_tweak_add(ctx, &new_pubkey, neg_scalar)) {
            std::cerr << "Failed to tweak the public key. The resulting key may be invalid.\n";
            continue;
        }

        // Serialize the new public key to compressed format
        unsigned char new_pubkey_serialized[33];
        size_t new_pubkey_len = 33;
        if(!secp256k1_ec_pubkey_serialize(ctx, new_pubkey_serialized, &new_pubkey_len, &new_pubkey, SECP256K1_EC_COMPRESSED)) {
            std::cerr << "Failed to serialize the new public key.\n";
            continue;
        }

        // Convert the new public key to hex
        std::string new_pubkey_hex = bytes_to_hex(new_pubkey_serialized, new_pubkey_len);

        // Convert the scalar to a BIGNUM and then to a decimal string for output
        BIGNUM* scalar_bn = BN_new();
        if (!scalar_bn || !BN_bin2bn(scalar, 32, scalar_bn)) {
            std::cerr << "Failed to convert scalar to BIGNUM.\n";
            if (scalar_bn) BN_free(scalar_bn);
            continue;
        }
        char* scalar_dec = BN_bn2dec(scalar_bn);
        std::string scalar_str(scalar_dec);
        BN_free(scalar_bn);
        OPENSSL_free(scalar_dec);

        // Write the new public key and corresponding recovery scalar to the output file
        outfile << new_pubkey_hex << " # " << scalar_str << "\n";

        // Optional: Display progress
        if((i + 1) % 1000 == 0 || (i + 1) == number_of_keys) {
            std::cout << "Generated " << (i + 1) << " / " << number_of_keys << " keys.\n";
        }
    }

    outfile.close();
    secp256k1_context_destroy(ctx);
    std::cout << "Successfully generated " << number_of_keys << " subtracted public keys and saved to " << output_file << "\n";

    return 0;
}
