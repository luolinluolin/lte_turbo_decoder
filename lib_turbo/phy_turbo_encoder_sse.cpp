/**********************************************************************
*
* <COPYRIGHT_TAG>
*
**********************************************************************/

/*
 * @file   phy_turbo_encoder_sse.cpp
 * @brief  turbo encoder
*/

#include <cstdint>

#include <immintrin.h>

#include "common_typedef_sdk.h"

#include "phy_turbo.h"
#include "phy_turbo_internal.h"
#include "gcc_inc.h"

#if defined(_BBLIB_SSE4_2_) || defined(_BBLIB_AVX2_) || defined(_BBLIB_AVX512_)

__align(64) uint16_t g_OutputWinTable8_sdk[256][8];
__align(64) uint8_t g_TailWinTable_sdk[8] = {0, 60, 200, 244, 160, 156, 104, 84};

/* const for 128-bit processing */
__m128i qp128 = _mm_setr_epi8(0xe4, 0x72, 0xb9, 0x5c,
                              0x2e, 0x97, 0xcb, 0xe5,
                              0x72, 0xb9, 0x5c, 0x2e,
                              0x97, 0xcb, 0xe5, 0x72);

__m128i shuffleMask = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

__m128i Mask64 = _mm_setr_epi8(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                               0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff);

__m128i g18 = _mm_setr_epi8(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xa0);

inline __m128i a_fun(__m128i cw)
{
    __m128i at1, at2, at3, at4, a;
    at1 = _mm_clmulepi64_si128(cw, qp128, 0x11);
    at2 = _mm_clmulepi64_si128(cw, qp128, 0x01);
    at3 = _mm_clmulepi64_si128(cw, qp128, 0x10);
    at4 = _mm_xor_si128(at2, at3);
    at4 = _mm_srli_si128(at4, 8);
    a = _mm_xor_si128(at1, at4);
    a = _mm_xor_si128(a, cw);
    return a;
}

#define unlikely_local(x)     __builtin_expect(!!(x), 0)

struct init_turbo_encoder_sse
{
    init_turbo_encoder_sse()
    {

    bblib_print_turbo_version();

    }
};

init_turbo_encoder_sse do_constructor_turbo_encoder_sse;

int32_t
bblib_lte_turbo_encoder_sse(const struct bblib_turbo_encoder_request *request,
    struct bblib_turbo_encoder_response *response)
{
    __align(64) uint8_t input_win_2[MAX_DATA_LEN_INTERLEAVE*4];

    bblib_lte_turbo_interleaver_8windows_sse(request->case_id, request->input_win,
                                            input_win_2);

    __m128i cw0, cw1, b0, b1, yt0, yt1, yr0, yr1, x0, x1, y0, y1;
    __m128i a0 = {0};
    __m128i a1 = {0};
    int32_t len128, lens, idx;
    int8_t bt0 = 0, bt1 = 0, bt2 = 0;
    int8_t b80, b81, yr80, yr81;
    uint32_t requestLength = request->length;
    // Add this line to fix klocwork SPECTRE.VARIANT1 warning
    if (requestLength >= (MAX_DATA_LEN_INTERLEAVE*4))
    {
        printf("Request length for turbo encoder out of range!\n");
        return -1;
    }

    uint8_t State0, State1, Tail0, Tail1, Tmp1, Tmp2;
    uint16_t TmpShort1, TmpShort2;

    len128 = requestLength & 0xFFFFFFF0;
    b0 = _mm_setzero_si128();
    yr0 = _mm_setzero_si128();
    b1 = _mm_setzero_si128();
    yr1 = _mm_setzero_si128();
    if (unlikely_local(len128 > 0))
    {
        for (idx = 0; idx < len128; idx+= 16)
        {
            cw0 = _mm_loadu_si128((__m128i *)(request->input_win + idx));
            cw1 = _mm_loadu_si128((__m128i *)(input_win_2 + idx));
            _mm_storeu_si128((__m128i *)(response->output_win_0 + idx), cw0);
            cw0 = _mm_shuffle_epi8 (cw0, shuffleMask);
            cw1 = _mm_shuffle_epi8 (cw1, shuffleMask);

            cw0 = _mm_xor_si128(cw0, b0);
            cw1 = _mm_xor_si128(cw1, b1);

            a0 = a_fun(cw0);
            a1 = a_fun(cw1);

            x0 = _mm_slli_si128(a0, 15);
            x1 = _mm_slli_si128(a1, 15);
            b0 = _mm_slli_epi64(x0, 1);
            b1 = _mm_slli_epi64(x1, 1);
            b0 = _mm_xor_si128(b0, x0);
            b1 = _mm_xor_si128(b1, x1);
            b0 = _mm_slli_epi64(b0, 5);
            b1 = _mm_slli_epi64(b1, 5);

            yt0 = _mm_clmulepi64_si128(a0, g18, 0x11);
            yt1 = _mm_clmulepi64_si128(a1, g18, 0x11);
            yt0 = _mm_xor_si128(yt0, a0);
            yt1 = _mm_xor_si128(yt1, a1);
            y0 = _mm_clmulepi64_si128(a0, g18, 0x10);
            y1 = _mm_clmulepi64_si128(a1, g18, 0x10);
            y0 = _mm_srli_si128(y0, 8);
            y1 = _mm_srli_si128(y1, 8);
            yt0 = _mm_xor_si128(yt0, y0);
            yt1 = _mm_xor_si128(yt1, y1);

            yt0 = _mm_xor_si128(yt0, yr0);
            yt1 = _mm_xor_si128(yt1, yr1);
            yt0 = _mm_shuffle_epi8(yt0, shuffleMask);
            yt1 = _mm_shuffle_epi8(yt1, shuffleMask);

            _mm_storeu_si128((__m128i *)(response->output_win_1 + idx), yt0);
            _mm_storeu_si128((__m128i *)(response->output_win_2 + idx), yt1);

            yr0 = _mm_slli_epi64(x0, 2);
            yr1 = _mm_slli_epi64(x1, 2);
            yr0 = _mm_xor_si128(x0, yr0);
            yr1 = _mm_xor_si128(x1, yr1);
            yr0 = _mm_slli_epi64(yr0, 5);
            yr1 = _mm_slli_epi64(yr1, 5);
        }
    }
    lens = requestLength - len128;
    if (unlikely_local(lens == 0))
    {
        /* tail processing */
        b80 = _mm_extract_epi8(b0, 15);
        yr80 = _mm_extract_epi8(yr0, 15);
        b81 = _mm_extract_epi8(b1, 15);
        yr81 = _mm_extract_epi8(yr1, 15);

        bt0 = bt1 = bt2 = 0;

        bt0 = bt0 | (b80 & 0x80);
        bt1 = bt1 | (yr80 & 0x80);
        bt2 = bt2 | ((b80 & 0x40) <<1);

        bt0 = bt0 | (yr80 & 0x40);
        bt1 = bt1 | ((b80 & 0x20) <<1);
        bt2 = bt2 | ((yr80 & 0x20) <<1);

        bt0 = bt0 | ((b81 & 0x80) >> 2);
        bt1 = bt1 | ((yr81 & 0x80) >> 2);
        bt2 = bt2 | ((b81 & 0x40) >> 1);

        bt0 = bt0 | ((yr81 & 0x40) >> 2);
        bt1 = bt1 | ((b81 & 0x20) >> 1);
        bt2 = bt2 | ((yr81 & 0x20) >> 1);

        *(response->output_win_0 + requestLength) = bt0;
        *(response->output_win_1 + requestLength) = bt1;
        *(response->output_win_2 + requestLength) = bt2;

        return 0;
    }
    else
    {
        if (lens >=64)
        {
            cw0 = _mm_loadu_si64(request->input_win + len128);
            cw1 = _mm_loadu_si64(input_win_2 + len128);
            _mm_storeu_si64(response->output_win_0 + len128, cw0);
            cw0 = _mm_slli_si128(cw0, 8);
            cw1 = _mm_slli_si128(cw1, 8);
            cw0 = _mm_shuffle_epi8 (cw0, shuffleMask);
            cw1 = _mm_shuffle_epi8 (cw1, shuffleMask);

            cw0 = _mm_xor_si128(cw0, b0);
            cw1 = _mm_xor_si128(cw1, b1);

            x0 = _mm_clmulepi64_si128(a0, qp128, 0x11);
            a0 = _mm_xor_si128(a0, x0);
            a0 = _mm_and_si128(a0, Mask64);

            x1 = _mm_clmulepi64_si128(a1, qp128, 0x11);
            a1 = _mm_xor_si128(a1, x1);
            a1 = _mm_and_si128(a1, Mask64);

            yt0 = _mm_clmulepi64_si128(a0, g18, 0x11);
            yt1 = _mm_clmulepi64_si128(a1, g18, 0x11);
            yt0 = _mm_xor_si128(yt0, a0);
            yt1 = _mm_xor_si128(yt1, a1);

            yt0 = _mm_xor_si128(yt0, yr0);
            yt1 = _mm_xor_si128(yt1, yr1);
            yt0 = _mm_shuffle_epi8(yt0, shuffleMask);
            yt1 = _mm_shuffle_epi8(yt1, shuffleMask);

            _mm_storeu_si64(response->output_win_1 + len128, yt0);
            _mm_storeu_si64(response->output_win_2 + len128, yt1);

            State0 = _mm_extract_epi8(a0, 8) & 0x7;
            State1 = _mm_extract_epi8(a1, 8) & 0x7;

            len128 += 64;
            lens = requestLength - len128;

            if (lens == 0)
            {
                /* tail processing */
                b80 = _mm_extract_epi8(b0, 15);
                yr80 = _mm_extract_epi8(yr0, 15);
                b81 = _mm_extract_epi8(b1, 15);
                yr81 = _mm_extract_epi8(yr1, 15);

                bt0 = bt1 = bt2 = 0;

                bt0 = bt0 | (b80 & 0x80);
                bt1 = bt1 | (yr80 & 0x80);
                bt2 = bt2 | ((b80 & 0x40) <<1);

                bt0 = bt0 | (yr80 & 0x40);
                bt1 = bt1 | ((b80 & 0x20) <<1);
                bt2 = bt2 | ((yr80 & 0x20) <<1);

                bt0 = bt0 | ((b81 & 0x80) >> 2);
                bt1 = bt1 | ((yr81 & 0x80) >> 2);
                bt2 = bt2 | ((b81 & 0x40) >> 1);

                bt0 = bt0 | ((yr81 & 0x40) >> 2);
                bt1 = bt1 | ((b81 & 0x20) >> 1);
                bt2 = bt2 | ((yr81 & 0x20) >> 1);

                *(response->output_win_0 + requestLength) = bt0;
                *(response->output_win_1 + requestLength) = bt1;
                *(response->output_win_2 + requestLength) = bt2;

                return 0;
            }
            else
            {
                State0 = _mm_extract_epi8(a0, 8) & 0x7;
                State1 = _mm_extract_epi8(a1, 8) & 0x7;
            }
        }
        else
        {
            State0 = _mm_extract_epi8(a0, 0) & 0x7;
            State1 = _mm_extract_epi8(a1, 0) & 0x7;
        }

        for(idx = len128; idx < requestLength; idx++)
        {
            Tmp1 = *(request->input_win + idx);
            Tmp2 = *(input_win_2 + idx);

            TmpShort1 = g_OutputWinTable8_sdk[Tmp1][State0];
            TmpShort2 = g_OutputWinTable8_sdk[Tmp2][State1];

            *(response->output_win_0 + idx) = Tmp1;
            *(response->output_win_1 + idx) = (uint8_t)(TmpShort1 >> 8);
            *(response->output_win_2 + idx) = (uint8_t)(TmpShort2 >> 8);

            State0 = (uint8_t)(TmpShort1 & 0x07);
            State1 = (uint8_t)(TmpShort2 & 0x07);
        }

        Tail0 = g_TailWinTable_sdk[State0];
        Tail1 = g_TailWinTable_sdk[State1];
        Tail1 = Tail1>>2;

        *(response->output_win_0 + requestLength) = (Tail0 & (128+64)) + (Tail1 & (32+16));
        Tail0 = Tail0 << 2;
        Tail1 = Tail1 << 2;
        *(response->output_win_1 + requestLength) = (Tail0 & (128+64)) + (Tail1 & (32+16));
        Tail0 = Tail0 << 2;
        Tail1 = Tail1 << 2;
        *(response->output_win_2 + requestLength) = (Tail0 & (128+64)) + (Tail1 & (32+16));

    }

    return 0;
}
#else
int32_t
bblib_lte_turbo_encoder_sse(const struct bblib_turbo_encoder_request *request,
    struct bblib_turbo_encoder_response *response)
{
    printf("bblib_turbo requires at least SSE4.2 ISA support to run\n");
    return(-1);
}
#endif
