// secp256k1_math.cu
#include "secp256k1_consts.cuh"
#include "secp256k1_math.cuh"  // ← ДОБАВИТЬ ЭТУ СТРОКУ
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Big256 as uint32_t[8] little-endian
typedef uint32_t Big256[8];
typedef uint32_t Big512[16];

// Point in Jacobian coordinates

// --- Big256 helpers ---
__device__ inline void big256_set_zero(Big256 a) { 
    for (int i = 0; i < 8; i++) a[i] = 0; 
}
__device__ inline void copy256(const Big256 src, Big256 dst) { 
    for (int i = 0; i < 8; i++) dst[i] = src[i]; 
}
__device__ inline int cmp256(const Big256 a, const Big256 b) {
    for (int i = 7; i >= 0; i--) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}
__device__ inline void add256(const Big256 a, const Big256 b, Big256 r) {
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t t = (uint64_t)a[i] + b[i] + carry;
        r[i] = (uint32_t)t;
        carry = t >> 32;
    }
}
__device__ inline void sub256(const Big256 a, const Big256 b, Big256 r) {
    uint64_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t av = (uint64_t)a[i];
        uint64_t bv = (uint64_t)b[i] + borrow;
        if (av >= bv) { r[i] = (uint32_t)(av - bv); borrow = 0; }
        else { r[i] = (uint32_t)((1ULL << 32) + av - bv); borrow = 1; }
    }
}
__device__ inline void mul256(const Big256 a, const Big256 b, Big512 c) {
    for (int i = 0; i < 16; i++) c[i] = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 8; j++) {
            uint64_t t = (uint64_t)a[i] * b[j] + c[i + j] + carry;
            c[i + j] = (uint32_t)t;
            carry = t >> 32;
        }
        c[i + 8] = (uint32_t)carry;
    }
}

// Modular reduction mod secp256k1 p
__device__ inline void reduce_secp256k1(const Big512 prod, Big256 out) {
    auto add_word = [](uint32_t& dst, uint64_t add, uint64_t& carry) {
        uint64_t s = (uint64_t)dst + add + carry;
        dst = (uint32_t)s;
        carry = s >> 32;
    };
    auto fold_once = [&](const Big512 in, Big512 out512) {
        Big256 H;
        for (int i = 0; i < 8; ++i) H[i] = in[8 + i];
        for (int i = 0; i < 16; ++i) out512[i] = 0;
        for (int i = 0; i < 8; ++i) out512[i] = in[i];
        uint64_t carry = 0;
        for (int i = 1; i < 16; ++i) {
            uint64_t add = (i - 1 < 8) ? (uint64_t)H[i - 1] : 0;
            add_word(out512[i], add, carry);
        }
        carry = 0;
        for (int i = 0; i < 8; ++i) {
            uint64_t add = (uint64_t)H[i] * 977ull;
            uint64_t s = (uint64_t)out512[i] + (uint32_t)add + carry;
            out512[i] = (uint32_t)s;
            carry = (s >> 32) + (add >> 32);
        }
        for (int i = 8; i < 16 && carry; ++i) {
            uint64_t s = (uint64_t)out512[i] + carry;
            out512[i] = (uint32_t)s;
            carry = s >> 32;
        }
    };
    Big512 t1, t2;
    fold_once(prod, t1);
    fold_once(t1, t2);
    for (int i = 0; i < 8; ++i) out[i] = t2[i];
    uint32_t P[8]; get_secp256k1_p(P);
    for (int k = 0; k < 2; ++k) {
        if (cmp256(out, P) >= 0) {
            Big256 tmp;
            sub256(out, P, tmp);
            copy256(tmp, out);
        } else break;
    }
}

// Modular mul & square
__device__ inline void modmul(const Big256 a, const Big256 b, Big256 r) {
    Big512 prod;
    mul256(a, b, prod);
    reduce_secp256k1(prod, r);
}
__device__ inline void modsquare(const Big256 a, Big256 r) { 
    modmul(a, a, r); 
}

// Modular inverse using Fermat's little theorem: a^(p-2) mod p
__device__ inline void modinv(const Big256 a, Big256 r) {
    Big256 acc;
    big256_set_zero(acc);
    acc[0] = 1;
    Big256 base;
    copy256(a, base);
    for (int wi = 7; wi >= 0; --wi) {
        uint32_t w = EXP_P_MINUS_2[wi];
        for (int bi = 31; bi >= 0; --bi) {
            modsquare(acc, acc);
            if ((w >> bi) & 1u) modmul(acc, base, acc);
        }
    }
    copy256(acc, r);
}

// Add/sub mod p
__device__ inline void addmod_p(const Big256 a, const Big256 b, Big256 r) {
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t t = (uint64_t)a[i] + b[i] + carry;
        r[i] = (uint32_t)t;
        carry = t >> 32;
    }
    uint32_t P[8]; get_secp256k1_p(P);
    if (carry || cmp256(r, P) >= 0) {
        sub256(r, P, r);
    }
}
__device__ inline void submod_p(const Big256 a, const Big256 b, Big256 r) {
    uint64_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t av = (uint64_t)a[i];
        uint64_t bv = (uint64_t)b[i] + borrow;
        if (av >= bv) { r[i] = (uint32_t)(av - bv); borrow = 0; }
        else { r[i] = (uint32_t)((1ULL << 32) + av - bv); borrow = 1; }
    }
    if (borrow) {
        uint32_t P[8]; get_secp256k1_p(P);
        add256(r, P, r);
    }
}

// Jacobian point operations
__device__ inline bool is_infty(const PointJ* p) {
    for (int i = 0; i < 8; i++) if (p->Z[i] != 0) return false;
    return true;
}
__device__ inline void to_jacobian(const Big256 x, const Big256 y, PointJ* out) {
    copy256(x, out->X); copy256(y, out->Y);
    big256_set_zero(out->Z); out->Z[0] = 1;
}
__device__ inline void from_jacobian(const PointJ* p, Big256 ax, Big256 ay) {
    if (is_infty(p)) { 
        big256_set_zero(ax); big256_set_zero(ay); 
        return; 
    }
    Big256 zinv; modinv(p->Z, zinv);
    Big256 zinv2; modsquare(zinv, zinv2);
    Big256 zinv3; modmul(zinv2, zinv, zinv3);
    modmul(p->X, zinv2, ax);
    modmul(p->Y, zinv3, ay);
}
__device__ inline void jacobian_double(const PointJ* p, PointJ* r_out) {
    if (is_infty(p)) { *r_out = *p; return; }
    Big256 Y2, XY2, X2, M, M2, Y4;
    modsquare(p->Y, Y2);
    modmul(p->X, Y2, XY2);
    modsquare(p->X, X2);
    Big256 threeX2; addmod_p(X2, X2, threeX2); addmod_p(threeX2, X2, M);
    Big256 S; addmod_p(XY2, XY2, S); addmod_p(S, S, S);
    modsquare(M, M2);
    modsquare(Y2, Y4);
    Big256 nx, twoS; addmod_p(S, S, twoS); submod_p(M2, twoS, nx);
    Big256 ny, S_minus_nx; submod_p(S, nx, S_minus_nx);
    Big256 nytmp; modmul(M, S_minus_nx, nytmp);
    Big256 eightY4; addmod_p(Y4, Y4, eightY4); addmod_p(eightY4, eightY4, eightY4);
    addmod_p(eightY4, eightY4, eightY4); submod_p(nytmp, eightY4, ny);
    Big256 nz, YZ; modmul(p->Y, p->Z, YZ); addmod_p(YZ, YZ, nz);
    copy256(nx, r_out->X); copy256(ny, r_out->Y); copy256(nz, r_out->Z);
}
__device__ inline void jacobian_add(const PointJ* p, const PointJ* q, PointJ* r_out) {
    if (is_infty(p)) { *r_out = *q; return; }
    if (is_infty(q)) { *r_out = *p; return; }
    Big256 Z2sq; modsquare(q->Z, Z2sq);
    Big256 U1; modmul(p->X, Z2sq, U1);
    Big256 Z1sq; modsquare(p->Z, Z1sq);
    Big256 U2; modmul(q->X, Z1sq, U2);
    Big256 Z2cu; modmul(Z2sq, q->Z, Z2cu);
    Big256 S1; modmul(p->Y, Z2cu, S1);
    Big256 Z1cu; modmul(Z1sq, p->Z, Z1cu);
    Big256 S2; modmul(q->Y, Z1cu, S2);
    if (cmp256(U1, U2) == 0) {
        if (cmp256(S1, S2) != 0) { big256_set_zero(r_out->X); big256_set_zero(r_out->Y); big256_set_zero(r_out->Z); return; }
        else { jacobian_double(p, r_out); return; }
    }
    Big256 H; submod_p(U2, U1, H);
    Big256 R; submod_p(S2, S1, R);
    Big256 H2; modsquare(H, H2);
    Big256 H3; modmul(H2, H, H3);
    Big256 U1H2; modmul(U1, H2, U1H2);
    Big256 R2; modsquare(R, R2);
    Big256 tmp1; submod_p(R2, H3, tmp1);
    Big256 twoU1H2; addmod_p(U1H2, U1H2, twoU1H2);
    Big256 nx; submod_p(tmp1, twoU1H2, nx);
    Big256 U1H2_minus_nx; submod_p(U1H2, nx, U1H2_minus_nx);
    Big256 Rmul; modmul(R, U1H2_minus_nx, Rmul);
    Big256 S1H3; modmul(S1, H3, S1H3);
    Big256 ny; submod_p(Rmul, S1H3, ny);
    Big256 nz; modmul(H, p->Z, nz); modmul(nz, q->Z, nz);
    copy256(nx, r_out->X); copy256(ny, r_out->Y); copy256(nz, r_out->Z);
}

// Scalar multiplication k*G
__device__ inline void scalar_mul(const Big256 scalar, PointJ* res) {
    Big256 Gx, Gy;
    for (int i = 0; i < 8; i++) { Gx[i] = Gx_const[i]; Gy[i] = Gy_const[i]; }
    PointJ G_jac;
    to_jacobian(Gx, Gy, &G_jac);
    big256_set_zero(res->X); big256_set_zero(res->Y); big256_set_zero(res->Z);
    PointJ R = *res;
    PointJ addp = G_jac;
    for (int wi = 7; wi >= 0; --wi) {
        uint32_t w = scalar[wi];
        for (int b = 31; b >= 0; --b) {
            jacobian_double(&R, &R);
            if ((w >> b) & 1u) jacobian_add(&R, &addp, &R);
        }
    }
    *res = R;
}

// --- New fast incremental search kernel ---
extern "C" __global__ void search_kernel_incremental(
    PointJ base_pub_jac_in,                 // начальный pub = d_min * G (Jacobian)
    const Big256 step,                      // шаг (обычно 1, но может быть любой)
    uint64_t start_offset,                  // с какого глобального шага начинаем
    uint64_t iterations_per_thread,         // сколько шагов делает каждый thread
    const uint8_t* d_target_x_prefix,       // префикс X-координаты
    int prefix_len,                         // длина префикса в байтах
    uint8_t target_prefix,                  // 0x02 или 0x03
    unsigned long long* d_matches           // счётчик совпадений
) {
    uint64_t thread_id = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total_threads = (uint64_t)gridDim.x * blockDim.x;

    // Локальная копия текущей точки
    PointJ pub = base_pub_jac_in;

    // Предвычисляем точку step*G один раз на thread
    PointJ step_point;
    scalar_mul(step, &step_point);

    // Пропускаем начальные шаги: pub += start_offset * (step*G)
    uint64_t offset = start_offset * total_threads + thread_id;
    PointJ temp = pub;
    for (int bit = 63; bit >= 0; --bit) {
        jacobian_double(&temp, &temp);
        if ((offset >> bit) & 1ULL) {
            jacobian_add(&temp, &step_point, &temp);
        }
    }
    pub = temp;

    // Основной цикл поиска
    for (uint64_t i = 0; i < iterations_per_thread; ++i) {
        // Конвертируем Jacobian → affine
        Big256 pub_x, pub_y;
        from_jacobian(&pub, pub_x, pub_y);

        // Формируем compressed pubkey: 0x02/0x03 + X (32 байта big-endian)
        uint8_t pubkey[33];
        pubkey[0] = (pub_y[0] & 1) ? 0x03 : 0x02;

        for (int w = 0; w < 8; ++w) {
            uint32_t word = pub_x[7 - w];
            pubkey[1 + w*4 + 0] = (word >> 24) & 0xFF;
            pubkey[1 + w*4 + 1] = (word >> 16) & 0xFF;
            pubkey[1 + w*4 + 2] = (word >> 8)  & 0xFF;
            pubkey[1 + w*4 + 3] = word & 0xFF;
        }

        // Быстрая проверка префикса
        if (pubkey[0] == target_prefix) {
            bool match = true;
            for (int b = 0; b < prefix_len; ++b) {
                if (pubkey[1 + b] != d_target_x_prefix[b]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                atomicAdd(d_matches, 1ULL);
            }
        }

        // Следующий ключ: pub += step*G
        jacobian_add(&pub, &step_point, &pub);
    }
}

// Kernel для вычисления base_pub = d_min * G один раз на GPU
__global__ void init_base_pub_kernel(const Big256 d_min, PointJ* out_base_pub) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        scalar_mul(d_min, out_base_pub);
    }
}
