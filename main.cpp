#include <cstdio>
#include <cstring>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "secp256k1_math.cuh"

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA ERROR %d: %s (line %d)\n", (int)err, cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

typedef uint32_t Big256[8];

int hex_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    return 0;
}

bool parse_hex_to_big256(const char* hex_str, Big256 out) {
    for (int i = 0; i < 8; i++) out[i] = 0;
    int len = 0;
    while (hex_str[len] && hex_str[len] != '\r' && hex_str[len] != '\n') len++;
    int bit_pos = 0;
    for (int i = len - 1; i >= 0; i--) {
        int v = hex_val(hex_str[i]);
        int word_idx = bit_pos / 32;
        int bit_in_word = bit_pos % 32;
        if (word_idx >= 8) return false;
        out[word_idx] |= (uint32_t)(v << bit_in_word);
        bit_pos += 4;
    }
    return true;
}

void add256_host(const Big256 a, const Big256 b, Big256 r) {
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t t = (uint64_t)a[i] + b[i] + carry;
        r[i] = (uint32_t)t;
        carry = t >> 32;
    }
}

void mul_uint64_host(const Big256 a, uint64_t b, Big256 r) {
    for (int i = 0; i < 8; i++) r[i] = 0;
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t low = (uint64_t)a[i] * (b & 0xFFFFFFFFULL);
        uint64_t high = (uint64_t)a[i] * (b >> 32);
        uint64_t t = low + carry;
        r[i] = (uint32_t)t;
        carry = (t >> 32) + high;
    }
    for (int i = 0; carry && i < 8; i++) {
        uint64_t t = (uint64_t)r[i] + (carry & 0xFFFFFFFFULL);
        r[i] = (uint32_t)t;
        carry = (t >> 32) + (carry >> 32);
    }
}

int main() {
    printf("=== secp256k1_search START ===\n");
    
    Big256 d_min, step;
    uint64_t num_iters = 0;
    uint8_t target_prefix = 0;
    uint8_t target_x_prefix[32] = {0};
    int prefix_len = 0;
    
    FILE* config = fopen("config.txt", "r");
    if (!config) {
        printf("ERROR: config.txt not found!\n");
        return 1;
    }
    
    printf("Reading config.txt...\n");
    char line[512];
    while (fgets(line, sizeof(line), config)) {
        char* eq = strchr(line, '=');
        if (!eq) continue;
        *eq = 0;
        char* key = line;
        char* val = eq + 1;
        while (*key == ' ' || *key == '\t') key++;
        int vlen = strlen(val);
        while (vlen > 0 && (val[vlen-1] == ' ' || val[vlen-1] == '\t' || 
               val[vlen-1] == '\r' || val[vlen-1] == '\n')) vlen--;
        val[vlen] = 0;
        
        if (strcmp(key, "target_prefix") == 0) {
            target_prefix = (uint8_t)strtoul(val, NULL, 16);
        } else if (strcmp(key, "pub_key_x") == 0) {
            prefix_len = 0;
            for (char* p = val; *p && prefix_len < 32; p += 2) {
                if (!((p[0] >= '0' && p[0] <= '9') || (p[0] >= 'a' && p[0] <= 'f') || (p[0] >= 'A' && p[0] <= 'F')) ||
                    !((p[1] >= '0' && p[1] <= '9') || (p[1] >= 'a' && p[1] <= 'f') || (p[1] >= 'A' && p[1] <= 'F'))) break;
                target_x_prefix[prefix_len] = (hex_val(p[0]) << 4) | hex_val(p[1]);
                prefix_len++;
            }
        } else if (strcmp(key, "d_min") == 0) {
            parse_hex_to_big256(val, d_min);
        } else if (strcmp(key, "step") == 0) {
            parse_hex_to_big256(val, step);
        } else if (strcmp(key, "num_iters") == 0) {
            num_iters = strtoull(val, NULL, 10);
        }
    }
    fclose(config);
    
    printf("Config: num_iters=%llu, prefix_len=%d, target=%02x\n", num_iters, prefix_len, target_prefix);
    if (num_iters == 0 || prefix_len == 0) {
        printf("ERROR: Invalid config\n");
        return 1;
    }
    
    const int threads_per_block = 256;
    const int max_batch_size = 1024 * 1024;  // 1M keys!
    int num_batches = (num_iters + max_batch_size - 1LL) / max_batch_size;
    
    printf("GPU setup: %d batches of %d keys\n", num_batches, max_batch_size);
    
    Big256* d_priv_batch;
    uint8_t* d_target_x;
    unsigned long long* d_matches;
    CUDA_CHECK(cudaMalloc(&d_priv_batch, max_batch_size * sizeof(Big256)));
    CUDA_CHECK(cudaMalloc(&d_target_x, 32));
    CUDA_CHECK(cudaMalloc(&d_matches, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_target_x, target_x_prefix, prefix_len, cudaMemcpyHostToDevice));
    
    unsigned long long h_total_matches = 0;
    auto t_start = std::chrono::high_resolution_clock::now();
    
    printf("Starting search...\n");
    for (int batch = 0; batch < num_batches; batch++) {
        uint64_t batch_start = (uint64_t)batch * max_batch_size;
        uint64_t batch_end = std::min(batch_start + max_batch_size, num_iters);
        int batch_size = (int)(batch_end - batch_start);
        
        // DYNAMIC MALLOC - работает с 1M!
        Big256* h_priv_batch = (Big256*)malloc(batch_size * sizeof(Big256));
        if (!h_priv_batch) {
            printf("ERROR: malloc failed for batch %d\n", batch);
            exit(1);
        }
        
        for (int i = 0; i < batch_size; i++) {
            uint64_t idx = batch_start + i;
            Big256 idx_step;
            mul_uint64_host(step, idx, idx_step);
            add256_host(d_min, idx_step, h_priv_batch[i]);
        }
        
        CUDA_CHECK(cudaMemcpy(d_priv_batch, h_priv_batch, batch_size * sizeof(Big256), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_matches, 0, sizeof(unsigned long long)));
        
        dim3 blocks((batch_size + threads_per_block - 1) / threads_per_block);
        dim3 threads(threads_per_block);
        search_pubkeys<<<blocks, threads>>>(d_priv_batch, target_prefix, d_target_x, prefix_len, d_matches);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        unsigned long long batch_matches;
        CUDA_CHECK(cudaMemcpy(&batch_matches, d_matches, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        h_total_matches += batch_matches;
        
        printf("Batch %d/%d: %llu matches\r", batch, num_batches, batch_matches);
        fflush(stdout);
        
        free(h_priv_batch);  // FREE MEMORY!
    }
    
    auto t_end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(t_end - t_start).count();
    
    printf("\n\n=== RESULTS ===\n");
    printf("Total iterations: %llu\n", num_iters);
    printf("Matches found: %llu\n", h_total_matches);
    printf("Speed: %.2f Mkeys/sec\n", num_iters / seconds / 1e6);
    printf("Total time: %.2f sec\n", seconds);
    
    CUDA_CHECK(cudaFree(d_priv_batch));
    CUDA_CHECK(cudaFree(d_target_x));
    CUDA_CHECK(cudaFree(d_matches));
    
    printf("=== DONE ===\n");
    return 0;
}
