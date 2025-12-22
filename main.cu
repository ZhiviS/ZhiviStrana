#include <cstdio>
#include <cstring>
#include <chrono>
#include <cstdint>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "secp256k1_math.cuh"

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error %d: %s (line %d)\n", (int)err, cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

typedef uint32_t Big256[8];

// Функция перевода hex символа в число
inline int hex_char_to_int(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    return -1;
}

// Парсинг hex в Big256 (little-endian)
bool parse_hex_to_big256(const char* hex, Big256 out) {
    memset(out, 0, 32);
    int len = 0;
    while (hex[len] && hex[len] != '\r' && hex[len] != '\n') len++;
    if (len > 64) len = 64;

    int byte_pos = 31;
    for (int i = len - 1; i >= 0; i -= 2) {
        if (byte_pos < 0) return false;

        int hi = hex_char_to_int(hex[i]);
        int lo = (i - 1 >= 0) ? hex_char_to_int(hex[i - 1]) : 0;
        if (hi < 0 || (i - 1 >= 0 && lo < 0)) return false;

        ((uint8_t*)out)[byte_pos--] = (uint8_t)((hi << 4) | lo);
        if (i > 0) i--;
    }
    return true;
}

int main() {
    printf("=== ZhiviStrana - Full GPU secp256k1 search ===\n");

    Big256 d_min = {0}, step = {0};
    uint64_t num_iters = 0;
    uint8_t target_prefix = 0;
    uint8_t target_x[32] = {0};
    int prefix_len = 0;

    FILE* f = fopen("config.txt", "r");
    if (!f) {
        printf("ERROR: config.txt not found!\n");
        return 1;
    }

    char line[512];
    while (fgets(line, sizeof(line), f)) {
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

        if (strcmp(key, "d_min") == 0) {
            parse_hex_to_big256(val, d_min);
        } else if (strcmp(key, "step") == 0) {
            parse_hex_to_big256(val, step);
        } else if (strcmp(key, "num_iters") == 0) {
            num_iters = strtoull(val, NULL, 0);
        } else if (strcmp(key, "target_prefix") == 0) {
            target_prefix = (uint8_t)strtoul(val, NULL, 16);
        } else if (strcmp(key, "pub_key_x") == 0) {
            prefix_len = 0;
            for (const char* p = val; *p && prefix_len < 32; p += 2) {
                int hi = hex_char_to_int(p[0]);
                int lo = hex_char_to_int(p[1]);
                if (hi < 0 || lo < 0) break;
                target_x[prefix_len++] = (uint8_t)((hi << 4) | lo);
            }
        }
    }
    fclose(f);

    if (num_iters == 0 || prefix_len == 0) {
        printf("ERROR: Invalid config parameters!\n");
        return 1;
    }

    printf("Config loaded:\n");
    printf("  num_iters:   %llu\n", num_iters);
    printf("  prefix_len:  %d bytes\n", prefix_len);

    const int BLOCKS = 180; //----------------------------------
    const int THREADS_PER_BLOCK = 512;
    const uint64_t STRIDE = (uint64_t)BLOCKS * THREADS_PER_BLOCK;
    const uint64_t ITERATIONS_PER_THREAD = 2000000ULL;

    // Вычисляем base_pub = d_min * G полностью на GPU
    PointJ* d_base_pub_jac;
    CUDA_CHECK(cudaMalloc(&d_base_pub_jac, sizeof(PointJ)));

    init_base_pub_kernel<<<1, 1>>>(d_min, d_base_pub_jac);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    uint8_t* d_target_x;
    unsigned long long* d_matches;
    CUDA_CHECK(cudaMalloc(&d_target_x, 32));
    CUDA_CHECK(cudaMalloc(&d_matches, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_target_x, target_x, 32, cudaMemcpyHostToDevice));

    uint64_t processed = 0;
    unsigned long long total_matches = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    printf("Starting full GPU search...\n");

    while (processed < num_iters) {
        uint64_t keys_this_round = std::min(STRIDE * ITERATIONS_PER_THREAD, num_iters - processed);
        uint64_t iters = keys_this_round / STRIDE;

        CUDA_CHECK(cudaMemset(d_matches, 0, sizeof(unsigned long long)));

        search_kernel_incremental<<<BLOCKS, THREADS_PER_BLOCK>>>(
            *d_base_pub_jac,       // <-- вот здесь dereference
            step,
            processed / STRIDE,
            iters,
            d_target_x,
            prefix_len,
            target_prefix,
            d_matches
        );


        CUDA_CHECK(cudaDeviceSynchronize());

        unsigned long long round_matches = 0;
        CUDA_CHECK(cudaMemcpy(&round_matches, d_matches, sizeof(round_matches), cudaMemcpyDeviceToHost));
        total_matches += round_matches;
        processed += keys_this_round;

        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        double speed = (elapsed > 0.0) ? processed / elapsed / 1e6 : 0.0;

        printf("Progress: %llu / %llu (%.1f%%) | Matches: %llu | Speed: %.2f Mkeys/sec\r",
               processed, num_iters, 100.0 * processed / num_iters, total_matches, speed);
        fflush(stdout);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();
    double final_speed = (total_time > 0.0) ? num_iters / total_time / 1e6 : 0.0;

    printf("\n\n=== RESULTS ===\n");
    printf("Total iterations: %llu\n", num_iters);
    printf("Matches found: %llu\n", total_matches);
    printf("Total time: %.2f sec\n", total_time);
    printf("Final speed: %.2f Mkeys/sec\n", final_speed);
    printf("=== DONE ===\n");

    cudaFree(d_target_x);
    cudaFree(d_matches);
	cudaFree(d_base_pub_jac);

    return 0;
}  // ← Эта скобка закрывает main()
   // ← И эта — весь файл (если нужно)
