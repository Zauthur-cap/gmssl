//
// Created by saint on 2021/12/9.
//

#include "sm3.h"
#include <arm_neon.h>
#include <string.h>
#include <thread>

// 寄存器初始值
#define GM_SM3_IV_A 0x7380166f
#define GM_SM3_IV_B 0x4914b2b9
#define GM_SM3_IV_C 0x172442d7
#define GM_SM3_IV_D 0xda8a0600
#define GM_SM3_IV_E 0xa96f30bc
#define GM_SM3_IV_F 0x163138aa
#define GM_SM3_IV_G 0xe38dee4d
#define GM_SM3_IV_H 0xb0fb0e4e

// Tj常量
#define GM_SM3_T_0 0x79CC4519
#define GM_SM3_T_1 0x7A879D8A

// FFj函数
#define GM_SM3_FF_0(x, y, z) ( (x) ^ (y) ^ (z) )
#define GM_SM3_FF_1(x, y, z) ( ( (x) & (y) ) | ( (x) & (z) ) | ( (y) & (z) ) )

// GGj函数
#define GM_SM3_GG_0(x, y, z) ( (x) ^ (y) ^ (z) )
#define GM_SM3_GG_1(x, y, z) ( ( (x) & (y) ) | ( (~(x)) & (z) ) )

// 循环左移
#define  GM_SM3_SHL(x, n) (((x) & 0xFFFFFFFF) << (n % 32))
#define GM_SM3_ROTL(x, n) (GM_SM3_SHL((x), n) | ((x) >> (32 - (n % 32))))

// P0 P1函数
#define GM_SM3_P_0(x) ((x) ^  GM_SM3_ROTL((x),9) ^ GM_SM3_ROTL((x),17))
#define GM_SM3_P_1(x) ((x) ^  GM_SM3_ROTL((x),15) ^ GM_SM3_ROTL((x),23))

// 字节转化为字
#ifndef GM_GET_UINT32_BE
#define GM_GET_UINT32_BE(n, b, i)                       \
{                                                       \
    (n) = ( (uint32_t) (b)[(i)    ] << 24 )             \
        | ( (uint32_t) (b)[(i) + 1] << 16 )             \
        | ( (uint32_t) (b)[(i) + 2] <<  8 )             \
        | ( (uint32_t) (b)[(i) + 3]       );            \
}
#endif

// 字转化为字节
#ifndef GM_PUT_UINT32_BE
#define GM_PUT_UINT32_BE(n, b ,i)                       \
{                                                       \
    (b)[(i)    ] = (unsigned char) ( (n) >> 24 );       \
    (b)[(i) + 1] = (unsigned char) ( (n) >> 16 );       \
    (b)[(i) + 2] = (unsigned char) ( (n) >>  8 );       \
    (b)[(i) + 3] = (unsigned char) ( (n)       );       \
}
#endif

// 消息扩展，消息Bi -> W


// 消息扩展，消息Bi -> W
static void gm_sm3_BiToW(const unsigned char* Bi, unsigned int* W) {
    int i;
    unsigned int tmp;

    // 使用 SIMD 加载数据
    uint8x16_t tmp_vec;
    tmp_vec = vld1q_u8(Bi);
    vst1q_u32(W, tmp_vec);

    for (i = 4; i <= 15; i++) {
        // 计算 tmp
        tmp = W[i - 16] ^ W[i - 9] ^ GM_SM3_ROTL(W[i - 3], 15);
        W[i] = GM_SM3_P_1(tmp) ^ (GM_SM3_ROTL(W[i - 13], 7)) ^ W[i - 6];
    }

    // 使用 SIMD 加载剩余的数据
    for (i = 16; i <= 63; i += 4) {
        uint32_t tmp_data[4];
        vst1q_u32(tmp_data, vld1q_u32(W + i - 16));

        // 计算 tmp
        tmp = tmp_data[0] ^ tmp_data[1] ^ GM_SM3_ROTL(tmp_data[2], 15);
        W[i] = GM_SM3_P_1(tmp) ^ (GM_SM3_ROTL(W[i - 13], 7)) ^ W[i - 6];
        W[i + 1] = W[i - 15] ^ W[i - 8] ^ GM_SM3_ROTL(W[i - 2], 15);
        W[i + 1] = GM_SM3_P_1(W[i + 1]) ^ (GM_SM3_ROTL(W[i - 12], 7)) ^ W[i - 5];
        W[i + 2] = W[i - 14] ^ W[i - 7] ^ GM_SM3_ROTL(W[i - 1], 15);
        W[i + 2] = GM_SM3_P_1(W[i + 2]) ^ (GM_SM3_ROTL(W[i - 11], 7)) ^ W[i - 4];
        W[i + 3] = W[i - 13] ^ W[i - 6] ^ GM_SM3_ROTL(W[i - 3], 15);
        W[i + 3] = GM_SM3_P_1(W[i + 3]) ^ (GM_SM3_ROTL(W[i - 10], 7)) ^ W[i - 3];
    }
}

// w 扩展算法
static inline void gm_sm3_WToW1(const unsigned int* W, unsigned int* W1) {
    int i;

    for (i = 0; i <= 59; i += 4) {
        uint32x4_t w_vec = vld1q_u32(W + i);
        uint32x4_t w4_vec = vld1q_u32(W + i + 4);
        uint32x4_t w1_vec = veorq_u32(w_vec, w4_vec);
        vst1q_u32(W1 + i, w1_vec);
    }

    // 处理剩余的数据
    for (i = 60; i <= 63; i++) {
        W1[i] = W[i] ^ W[i + 4];
    }
}



// 压缩算法
static void gm_sm3_CF(const unsigned int * W, const unsigned int * W1, gm_sm3_context * ctx)
{
    unsigned int SS1;
    unsigned int SS2;
    unsigned int TT1;
    unsigned int TT2;
    unsigned int A, B, C, D, E, F, G, H;
    unsigned int Tj;
    int j;

    // ABCDEFGH = V (i)
    A = ctx->state[0];
    B = ctx->state[1];
    C = ctx->state[2];
    D = ctx->state[3];
    E = ctx->state[4];
    F = ctx->state[5];
    G = ctx->state[6];
    H = ctx->state[7];

    for(j = 0; j < 64; j++)
    {
        if(j < 16)
        {
            // if 0 <= j <= 15 Tj = 0x79cc4519
            Tj = GM_SM3_T_0;
        }
        else
        {
            // if j > 15 Tj = 0x7a879d8a
            Tj = GM_SM3_T_1;
        }
        // SS1 = ((A <<< 12) + E + (Tj <<< j)) <<< 7
        SS1 = GM_SM3_ROTL((GM_SM3_ROTL(A, 12) + E + GM_SM3_ROTL(Tj, j)), 7);
        // SS2 = SS1 ^ (A <<< 12)
        SS2 = SS1 ^ GM_SM3_ROTL(A, 12);

        // TT1 = FFj(A, B, C) + D + SS2 + Wj1
        // TT2 = GGj(E, F, G) + H + SS1 + Wj
        if(j < 16)
        {
            TT1 = GM_SM3_FF_0(A, B, C) + D + SS2 + W1[j];
            TT2 = GM_SM3_GG_0(E, F, G) + H + SS1 + W[j];
        }
        else
        {
            TT1 = GM_SM3_FF_1(A, B, C) + D + SS2 + W1[j];
            TT2 = GM_SM3_GG_1(E, F, G) + H + SS1 + W[j];
        }

        // D = C
        D = C;
        // C = B <<< 9
        C = GM_SM3_ROTL(B, 9);
        // B = A
        B = A;
        // A = TT1
        A = TT1;
        // H = G
        H = G;
        // G = F <<< 19
        G = GM_SM3_ROTL(F, 19);
        // F = E
        F = E;
        // E = P0(TT2)
        E = GM_SM3_P_0(TT2);
    }

    // V(i+1) = ABCDEFGH ^ V(i)
    ctx->state[0] ^= A;
    ctx->state[1] ^= B;
    ctx->state[2] ^= C;
    ctx->state[3] ^= D;
    ctx->state[4] ^= E;
    ctx->state[5] ^= F;
    ctx->state[6] ^= G;
    ctx->state[7] ^= H;
}

// 压缩算法
static void gm_sm3_compress(gm_sm3_context * ctx) {
    unsigned int W[68];
    unsigned int W1[64];

    // Bi 扩展为 W
    gm_sm3_BiToW(ctx->buf, W);

    // W 扩展为 W1
    gm_sm3_WToW1(W, W1);

    // 压缩
    gm_sm3_CF(W, W1, ctx);
}

/**
 * 摘要算法初始化
 * @param ctx 上下文
 */
void gm_sm3_init(gm_sm3_context * ctx) {
    ctx->state[0] = GM_SM3_IV_A;
    ctx->state[1] = GM_SM3_IV_B;
    ctx->state[2] = GM_SM3_IV_C;
    ctx->state[3] = GM_SM3_IV_D;
    ctx->state[4] = GM_SM3_IV_E;
    ctx->state[5] = GM_SM3_IV_F;
    ctx->state[6] = GM_SM3_IV_G;
    ctx->state[7] = GM_SM3_IV_H;
    ctx->cur_buf_len = 0;
    ctx->compressed_len = 0;
}

/**
 * 添加消息
 * @param ctx 上下文
 * @param input 消息
 * @param iLen 消息长度（字节）
 */
void gm_sm3_update(gm_sm3_context* ctx, const unsigned char* input, unsigned int iLen)
{
    // 计算剩余数据长度
    unsigned int remaining_len = 64 - ctx->cur_buf_len;

    // 如果输入数据不足以填满当前数据块，直接拷贝到缓冲区
    if (iLen < remaining_len)
    {
        memcpy(ctx->buf + ctx->cur_buf_len, input, iLen);
        ctx->cur_buf_len += iLen;
        return;
    }

    // 拷贝剩余数据到缓冲区
    memcpy(ctx->buf + ctx->cur_buf_len, input, remaining_len);
    ctx->cur_buf_len += remaining_len;

    // 压缩当前数据块
    gm_sm3_compress_inline((const unsigned int*)ctx->buf, nullptr, ctx);
    ctx->compressed_len += 512;
    ctx->cur_buf_len = 0;

    // 使用多线程并行处理剩余数据
    unsigned int remaining_bytes = iLen - remaining_len;
    unsigned int num_threads = std::thread::hardware_concurrency(); // 获取可用线程数
    std::vector<std::thread> threads(num_threads);

    for (unsigned int i = 0; i < num_threads; i++)
    {
        unsigned int thread_bytes = remaining_bytes / num_threads;
        if (i < (remaining_bytes % num_threads))
            thread_bytes++;

        threads[i] = std::thread(gm_sm3_compress_thread, ctx, input + remaining_len, thread_bytes);
        remaining_len += thread_bytes;
    }

    for (auto& thread : threads)
    {
        thread.join();
    }
}

static const unsigned char gm_sm3_padding[64] = {
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

/**
 * 计算摘要
 * @param ctx 上下文
 * @param output 输出摘要结果
 */
void gm_sm3_done(gm_sm3_context * ctx, unsigned char output[32]) {
    uint32_t padn;
    unsigned char msglen[8];
    uint64_t total_len, high, low;

    // 消息的总长度(比特) = 剩余未压缩数据的长度(字节) * 8
    total_len = ctx->compressed_len + (ctx->cur_buf_len << 3);
    high = (total_len >> 32) & 0x0FFFFFFFF;
    low = total_len & 0x0FFFFFFFF;

    GM_PUT_UINT32_BE(high, msglen, 0);
    GM_PUT_UINT32_BE(low,  msglen, 4);

    // 计算填充长度，因为事先要添加一比特，故应计算cur_buf_len + 1是否超过56
    padn = ((ctx->cur_buf_len + 1) <= 56) ? (56 - ctx->cur_buf_len) : (120 - ctx->cur_buf_len);

    // 添加填充
    gm_sm3_update(ctx, (unsigned char *) gm_sm3_padding, padn);
    gm_sm3_update(ctx, msglen, 8);

    // output
    GM_PUT_UINT32_BE(ctx->state[0], output,  0);
    GM_PUT_UINT32_BE(ctx->state[1], output,  4);
    GM_PUT_UINT32_BE(ctx->state[2], output,  8);
    GM_PUT_UINT32_BE(ctx->state[3], output, 12);
    GM_PUT_UINT32_BE(ctx->state[4], output, 16);
    GM_PUT_UINT32_BE(ctx->state[5], output, 20);
    GM_PUT_UINT32_BE(ctx->state[6], output, 24);
    GM_PUT_UINT32_BE(ctx->state[7], output, 28);
}

/**
 * 直接计算消息的摘要
 * @param input 消息
 * @param iLen 消息长度（字节）
 * @param output 输出摘要结果
 */
void gm_sm3(const unsigned char * input, unsigned int iLen, unsigned char output[32]) {
    gm_sm3_context ctx;
    gm_sm3_init(&ctx);
    gm_sm3_update(&ctx, input, iLen);
    gm_sm3_done(&ctx, output);
}

