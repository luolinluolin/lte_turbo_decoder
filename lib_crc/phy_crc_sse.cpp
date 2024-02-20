/**********************************************************************
*
* <COPYRIGHT_TAG>
*
**********************************************************************/

/**
 * @file   phy_crc_sse.cpp
 * @brief  Implementation of LTE CRC24A/CRC24B and the corresponding
 *         CRC generation, check functions
 */

/**
 * Include public/global header files
 */
#include <immintrin.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "phy_crc_internal.h"
#include "phy_crc.h"
#if defined(_BBLIB_SSE4_2_) || defined(_BBLIB_AVX2_) || defined(_BBLIB_AVX512_)
struct init_crc_sse
{
    init_crc_sse()
    {

        bblib_print_crc_version();

    }
};

init_crc_sse do_constructor_crc_sse;

static const __m128i k_shift_mask = _mm_set_epi8(0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                                 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff);
static const __m128i k_shuf_mask = _mm_set_epi8(0x03, 0x02, 0x01, 0x00, 0x03, 0x02, 0x01, 0x00,
                                                0x03, 0x02, 0x01, 0x00, 0x03, 0x02, 0x01, 0x00);
static const __m128i k_64 = _mm_set_epi8(0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                         0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40);
static const __m128i k_endian_shuf_mask32 = _mm_set_epi8(0x00, 0x01, 0x02, 0x03, 0x00, 0x01, 0x02, 0x03,
                                                         0x00, 0x01, 0x02, 0x03, 0x00, 0x01, 0x02, 0x03);
const static __m128i k_endian_shuf_mask128= _mm_set_epi8(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                                                        0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);

void bblib_lte_crc24a_gen_sse(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    uint8_t *data = request->data;
    uint8_t *dataOut = response->data;

    // len is passed in as bits, so turn into bytes
    uint32_t len_bytes = request->len / 8;

    /* record the start of input data */
    int32_t    i = 0;
    /* A= B mod C => A*K = B*K mod C*K, set K = 2^8, then compute CRC-32, the most significant */
    /* 24 bits is final 24 bits CRC. */
    const static uint64_t  CRC24APOLY = 0x1864CFB;   //CRC-24A polynomial
    const static uint64_t  CRC24APLUS8 = CRC24APOLY << 8;

    /* some pre-computed key constants */
    const static uint32_t k192   = 0x2c8c9d00;   //t=128+64, x^192 mod CRC24APLUS8, verified
    const static uint32_t k128   = 0x64e4d700;   //t=128, x^128 mod CRC24APLUS8, verified
    const static uint32_t k96    = 0xfd7e0c00;   //t=96, x^96 mod CRC24APLUS8, verified
    const static uint32_t k64    = 0xd9fe8c00;   //t=64, x^64 mod CRC24APLUS8, verified
    const static uint64_t u      = 0x1f845fe24;  //u for crc24A * 256, floor(x^64 / CRC24APLUS8), verified
    const static __m128i ENDIA_SHUF_MASK = _mm_set_epi8(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                                                  0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);

    __m128i xmm3, xmm2, xmm1, xmm0;

    /* 1. fold by 128bit. remaining length <=2*128bits. */
    xmm3 = _mm_set_epi32(0, k192, 0, k128);
    xmm1 = _mm_load_si128((__m128i *)data); data += 16;
    xmm1 = _mm_shuffle_epi8(xmm1, ENDIA_SHUF_MASK);

    for (i=(len_bytes>>4)-1; i>0; i--){
        xmm2 = xmm1;
        xmm0 = _mm_load_si128((__m128i *)data);  data += 16;
        xmm0 = _mm_shuffle_epi8(xmm0, ENDIA_SHUF_MASK);
        xmm1 = _mm_clmulepi64_si128(xmm1, xmm3, 0x00);
        xmm2 = _mm_clmulepi64_si128(xmm2, xmm3, 0x11);
        xmm0 = _mm_xor_si128(xmm1, xmm0);
        xmm1 = _mm_xor_si128(xmm2, xmm0);
    }


    /* 2. if remaining length > 128 bits, then pad zero to the most-significant bit to grow to 256bits length,
     * then fold once to 128 bits. */
    if (len_bytes>16){
        xmm0 = _mm_load_si128((__m128i *)data); //load remaining len%16 bytes and maybe some garbage bytes.
        xmm0 = _mm_shuffle_epi8(xmm0, ENDIA_SHUF_MASK);
        for (i=15-len_bytes%16; i>=0; i--){
            xmm0 = _mm_srli_si128(xmm0, 1);
            xmm0 = _mm_insert_epi8(xmm0, _mm_extract_epi8(xmm1, 0), 15);
            xmm1 = _mm_srli_si128(xmm1, 1);
        }
        xmm2 = xmm1;
        xmm1 = _mm_clmulepi64_si128(xmm1, xmm3, 0x00);
        xmm2 = _mm_clmulepi64_si128(xmm2, xmm3, 0x11);
        xmm0 = _mm_xor_si128(xmm1, xmm0);
        xmm0 = _mm_xor_si128(xmm2, xmm0);
    }
    else{
        xmm0 = xmm1;
        for (i=16-len_bytes; i>0; i--){
            xmm0 = _mm_srli_si128(xmm0, 1);
        }
    }


    /* 3. Apply 64 bits fold to 64 bits + 32 bits crc(32 bits zero) */
    xmm3 =  _mm_set_epi32(0, 0, 0, k96);
    xmm1 = _mm_clmulepi64_si128(xmm0, xmm3, 0x01);
    xmm2 = _mm_slli_si128(xmm0, 8);
    xmm2 = _mm_srli_si128(xmm2, 4);
    xmm0 = _mm_xor_si128(xmm1, xmm2);


    /* 4. Apply 32 bits fold to 32 bits + 32 bits crc(32 bits zero) */
    xmm3 =  _mm_set_epi32(0, 0, 0, k64);
    xmm1 = _mm_clmulepi64_si128(xmm0, xmm3, 0x01);
    xmm0 = _mm_slli_si128(xmm0, 8);
    xmm0 = _mm_srli_si128(xmm0, 8);
    xmm0 = _mm_xor_si128(xmm1, xmm0);

    /* 5. Use Barrett Reduction Algorithm to calculate the 32 bits crc.
     * Output: C(x)  = R(x) mod P(x)
     * Step 1: T1(x) = floor(R(x)/x^32)) * u
     * Step 2: T2(x) = floor(T1(x)/x^32)) * P(x)
     * Step 3: C(x)  = R(x) xor T2(x) mod x^32 */
    xmm1 = _mm_set_epi32(0, 0, 1, (u & 0xFFFFFFFF));
    xmm2 = _mm_srli_si128(xmm0, 4);
    xmm1 = _mm_clmulepi64_si128(xmm1, xmm2, 0x00);
    xmm1 = _mm_srli_si128(xmm1, 4);
    xmm2 = _mm_set_epi32(0, 0, 1, (CRC24APLUS8 & 0xFFFFFFFF));
    xmm1 = _mm_clmulepi64_si128(xmm1, xmm2, 0x00);
    xmm0 = _mm_xor_si128(xmm0, xmm1);
    xmm1 = _mm_set_epi32(0, 0, 0, 0xFFFFFFFF);
    xmm0 = _mm_and_si128 (xmm0, xmm1);


    /* 6. Update Result */
    /* add crc to last 3 bytes. */
    dataOut[len_bytes]   =  _mm_extract_epi8(xmm0, 3);
    dataOut[len_bytes+1] =  _mm_extract_epi8(xmm0, 2);
    dataOut[len_bytes+2] =  _mm_extract_epi8(xmm0, 1);
    response->len = (len_bytes + 3)*8;

    /* the most significant 24 bits of the 32 bits crc is the finial 24 bits crc. */
    response->crc_value = (((uint32_t)_mm_extract_epi32(xmm0, 0)) >> 8);
}



void bblib_lte_crc24b_gen_sse(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    uint8_t *data = request->data;
    uint8_t *dataOut = response->data;

    // len is passed in as bits, so turn into bytes
    uint32_t len_bytes = request->len / 8;

    /* record the start of input data */
    int32_t    i = 0;

    /* A= B mod C => A*K = B*K mod C*K, set K = 2^8, then compute CRC-32, the most significant
     * 24 bits is final 24 bits CRC. */
    const static uint64_t  CRC24BPOLY = 0x1800063;    //CRC24B Polynomial
    const static uint64_t  CRC24BPLUS8 = CRC24BPOLY << 8;

    /* some pre-computed key constants */
    const static uint32_t k192   = 0x42000100;   //t=128+64, x^192 mod CRC24BPLUS8, verified
    const static uint32_t k128   = 0x80140500;   //t=128, x^128 mod CRC24BPLUS8, verified
    const static uint32_t k96    = 0x09000200;   //t=96, x^96 mod CRC24BPLUS8, verified
    const static uint32_t k64    = 0x90042100;   //t=64, x^64 mod CRC24BPLUS8, verified
    const static uint64_t u      = 0x1ffff83ff;  //u for crc24A * 256, floor(x^64 / CRC24BPULS8), verified
    const static __m128i ENDIA_SHUF_MASK = _mm_set_epi8(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                                                  0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
    /* variables */
    __m128i xmm3, xmm2, xmm1, xmm0;

    /* 1. fold by 128bit. remaining length <=2*128bits. */
    xmm3 = _mm_set_epi32(0, k192, 0, k128);
    xmm1 = _mm_load_si128((__m128i *)data); data += 16;
    xmm1 = _mm_shuffle_epi8(xmm1, ENDIA_SHUF_MASK);

    for (i=(len_bytes>>4)-1; i>0; i--){
        xmm2 = xmm1;
        xmm0 = _mm_load_si128((__m128i *)data);  data += 16;
        xmm0 = _mm_shuffle_epi8(xmm0, ENDIA_SHUF_MASK);
        xmm1 = _mm_clmulepi64_si128(xmm1, xmm3, 0x00);
        xmm2 = _mm_clmulepi64_si128(xmm2, xmm3, 0x11);
        xmm0 = _mm_xor_si128(xmm1, xmm0);
        xmm1 = _mm_xor_si128(xmm2, xmm0);
    }

    /* 2. if remaining length > 128 bits, then pad zero to the most-significant bit to grow to 256bits length,
     *   then fold once to 128 bits. */
    if (16 < len_bytes){
        xmm0 = _mm_load_si128((__m128i *)data); //load remaining len%16 bytes and maybe some garbage bytes.
        xmm0 = _mm_shuffle_epi8(xmm0, ENDIA_SHUF_MASK);
        for (i=15-len_bytes%16; i>=0; i--){
            xmm0 = _mm_srli_si128(xmm0, 1);
            xmm0 = _mm_insert_epi8(xmm0, _mm_extract_epi8(xmm1, 0), 15);
            xmm1 = _mm_srli_si128(xmm1, 1);
        }
        xmm2 = xmm1;
        xmm1 = _mm_clmulepi64_si128(xmm1, xmm3, 0x00);
        xmm2 = _mm_clmulepi64_si128(xmm2, xmm3, 0x11);
        xmm0 = _mm_xor_si128(xmm1, xmm0);
        xmm0 = _mm_xor_si128(xmm2, xmm0);
    }
    else{
        xmm0 = xmm1;
        for (i=16-len_bytes; i>0; i--){
            xmm0 = _mm_srli_si128(xmm0, 1);
        }
    }

    /* 3. Apply 64 bits fold to 64 bits + 32 bits crc(32 bits zero) */
    xmm3 =  _mm_set_epi32(0, 0, 0, k96);
    xmm1 = _mm_clmulepi64_si128(xmm0, xmm3, 0x01);
    xmm2 = _mm_slli_si128(xmm0, 8);
    xmm2 = _mm_srli_si128(xmm2, 4);
    xmm0 = _mm_xor_si128(xmm1, xmm2);

    /* 4. Apply 32 bits fold to 32 bits + 32 bits crc(32 bits zero) */
    xmm3 =  _mm_set_epi32(0, 0, 0, k64);
    xmm1 = _mm_clmulepi64_si128(xmm0, xmm3, 0x01);
    xmm0 = _mm_slli_si128(xmm0, 8);
    xmm0 = _mm_srli_si128(xmm0, 8);
    xmm0 = _mm_xor_si128(xmm1, xmm0);

    /* 5. Use Barrett Reduction Algorithm to calculate the 32 bits crc.
     * Output: C(x)  = R(x) mod P(x)
     * Step 1: T1(x) = floor(R(x)/x^32)) * u
     * Step 2: T2(x) = floor(T1(x)/x^32)) * P(x)
     * Step 3: C(x)  = R(x) xor T2(x) mod x^32 */
    xmm1 = _mm_set_epi32(0, 0, 1, (u & 0xFFFFFFFF));
    xmm2 = _mm_srli_si128(xmm0, 4);
    xmm1 = _mm_clmulepi64_si128(xmm1, xmm2, 0x00);
    xmm1 = _mm_srli_si128(xmm1, 4);
    xmm2 =  _mm_set_epi32(0, 0, 1, (CRC24BPLUS8 & 0xFFFFFFFF));
    xmm1 = _mm_clmulepi64_si128(xmm1, xmm2, 0x00);
    xmm0 = _mm_xor_si128(xmm0, xmm1);
    xmm1 = _mm_set_epi32(0, 0, 0, 0xFFFFFFFF);
    xmm0 = _mm_and_si128 (xmm0, xmm1);


    /* 6. Update Result */
    /* add crc to last 3 bytes. */
    dataOut[len_bytes]   =  _mm_extract_epi8(xmm0, 3);
    dataOut[len_bytes+1] =  _mm_extract_epi8(xmm0, 2);
    dataOut[len_bytes+2] =  _mm_extract_epi8(xmm0, 1);
    response->len = (len_bytes + 3)*8;

    /* the most significant 24 bits of the 32 bits crc is the finial 24 bits crc. */
    response->crc_value = (((uint32_t)_mm_extract_epi32(xmm0, 0)) >> 8);

}



void bblib_lte_crc24a_check_sse(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    if (request->data == NULL){
        printf("bblib_lte_crc24a_check input / output address error \n");
        response->check_passed = false;
        return;
    }

    // len is passed in as bits, so turn into bytes
    uint32_t len_bytes = request->len / 8;

    /* CRC in the original sequence */
    uint32_t CRC_orig = 0;

    CRC_orig = ((request->data[len_bytes]<<16)&0x00FF0000) +
               ((request->data[len_bytes+1]<<8)&0x0000FF00) +
               (request->data[len_bytes+2]&0x000000FF);

    bblib_lte_crc24a_gen_sse(request, response);

    if (response->crc_value != CRC_orig)
        response->check_passed = false;
    else
        response->check_passed = true;
}



void bblib_lte_crc24b_check_sse(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    if (request->data == NULL){
        printf("bblib_lte_crc24b_check input / output address error \n");
        response->check_passed = false;
        return;
    }

    // len is passed in as bits, so turn into bytes
    uint32_t len_bytes = request->len / 8;

    /* CRC in the original sequence */
    uint32_t CRC_orig = 0;

    CRC_orig = ((request->data[len_bytes]<<16)&0x00FF0000) +
               ((request->data[len_bytes+1]<<8)&0x0000FF00) +
               (request->data[len_bytes+2]&0x000000FF);
    bblib_lte_crc24b_gen_sse(request, response);

    if (response->crc_value != CRC_orig)
        response->check_passed = false;
    else
        response->check_passed = true;
}

void mask_make_128i(__mmask16 k, uint8_t *mask_v)
{
    for (uint8_t j = 0; j < 16; j++)
    {
        __mmask16 mask = 0x0001;
        mask_v[j] = (k & (mask << j)) ? 128 : 0;
    }
}

__m128i bit_shift_right_m128i_sse(__m128i data, uint32_t shift_bits)
{
    // const auto src0 = _mm_rorv_epi64(data, count); // roate right shift_bits per 64 bits
    // SSE: _mm_rorv_epi64
    uint64_t initMask = 0xFFFFFFFFFFFFFFFF;
    uint64_t rotateMask = initMask >> (64 - shift_bits);
    __m128i rotMask = _mm_set_epi64x(rotateMask, rotateMask);
    __m128i rotData = _mm_and_si128 (data, rotMask); // get low 24bits per 64 bits
    rotData = _mm_slli_epi64(rotData, (64 - shift_bits));  // left shift to occupy high 24 bits

    uint64_t dataMask = ~rotateMask;
    __m128i datMask = _mm_set_epi64x(dataMask, dataMask); 
    __m128i data0 = _mm_and_si128 (data, datMask);      // get high 40 bits per 64 bits
    __m128i data1 = _mm_srli_epi64(data0, shift_bits); //right shift 24 bits
    __m128i src0 = _mm_or_si128 (rotData, data1); // rotate high 24 bits + low 40 bits
    // end SSE _mm_rorv_epi64

    __m128i src1 = _mm_srli_si128(src0, 8); // right shift 8 bytes (64 bits), only use high 64 bits
    // Blend src0 & src1 together based on mask. imm 0xd8 is pre-determined for blend logic table
    __m128i count = _mm_set_epi32(0, shift_bits, 0, shift_bits);
    __m128i temp = _mm_sub_epi64(k_64, count); // 64 - 24 = 40, only left 40 bits is effective
    __m128i mask_r = _mm_sll_epi64(k_shift_mask, temp); //k_shift_mask 0xFF left shift 40 (0000 0000 00) per 64, 16F leave 6F (FFFF FF00 0000 0000)

    // const auto result = _mm_ternarylogic_epi64(src0, src1, mask_r, 0xd8);
    // SSE: _mm_ternarylogic_epi64(A, B, C, imm8)
    __m128i t0 = _mm_and_si128(mask_r, src1);
    __m128i t1 = _mm_andnot_si128(mask_r, src0);
    __m128i t2 = _mm_or_si128(t0, t1);
    return t2;
}

__m128i shift_right_256_sse(__m128i data, __m128i next_data, uint32_t shift_bits)
{
    uint64_t initMask = 0xFFFFFFFFFFFFFFFF;
    uint32_t left_bits = 64 - shift_bits;
    uint64_t rotateMask = initMask >> left_bits;
    __m128i rotMask = _mm_set_epi64x(rotateMask, rotateMask);
    __m128i rotData = _mm_and_si128 (data, rotMask);
    rotData = _mm_slli_epi64(rotData, left_bits);

    uint64_t dataMask = ~rotateMask;
    __m128i datMask = _mm_set_epi64x(dataMask, dataMask); 
    __m128i data0 = _mm_and_si128 (data, datMask);     
    __m128i data1 = _mm_srli_epi64(data0, shift_bits);

    __m128i src0 = _mm_or_si128 (rotData, data1);
    const __m128i src1 = _mm_slli_si128(src0, 8);

    // Shift right next data in m128 register by required amount (src0)
    // const __m128i src2 = bit_shift_right_m128i_sse(next_data, shift_bits);
    /*******************************************************/
    __m128i next_rotData = _mm_and_si128 (next_data, rotMask);
    next_rotData = _mm_slli_epi64(next_rotData, left_bits);

    __m128i next_data0 = _mm_and_si128 (next_data, datMask);
    __m128i next_data1 = _mm_srli_epi64(next_data0, shift_bits);

    __m128i next_src0 = _mm_or_si128 (next_rotData, next_data1);
    __m128i next_src1 = _mm_srli_si128(next_src0, 8);
    __m128i count = _mm_set_epi32(0, shift_bits, 0, shift_bits);
    const __m128i mask_shift_r = _mm_sub_epi64(k_64, count);
    __m128i next_mask_r = _mm_sll_epi64(k_shift_mask, mask_shift_r);

    // const auto result = _mm_ternarylogic_epi64(src0, src1, mask_r, 0xd8);
    // SSE: _mm_ternarylogic_epi64(A, B, C, imm8)
    __m128i next_t0 = _mm_and_si128(next_mask_r, next_src1);
    __m128i next_t1 = _mm_andnot_si128(next_mask_r, next_src0);
    __m128i next_t2 = _mm_or_si128(next_t0, next_t1);
    const __m128i src2 = next_t2;
    /******************************************************/
    const __m128i k_shift_mask_temp = _mm_set_epi64x(0xFFFFFFFFFFFFFFFF, 0);
    const __m128i mask_r = _mm_sll_epi64(k_shift_mask_temp, mask_shift_r);

    // Blend src0 & src1 together based on mask. imm 0xd8 is pre-determined for blend logic table
    // const auto result = _mm_ternarylogic_epi64(src0, src1, mask_r, 0xd8);
    // SSE: _mm_ternarylogic_epi64(A, B, C, imm8)
    __m128i t0 = _mm_and_si128(mask_r, src1);    //B
    __m128i t1 = _mm_andnot_si128(mask_r, src2); //A
    __m128i t2 = _mm_or_si128(t0, t1);
    return t2;
}

struct crc24c_1_parameter
{
    const static uint64_t CRC24CPOLY = 0x1B2B117;      //CRC24C polynomial
    //Shift CRCPOLY by 32 minus 24bits since using a 24bit polynomial
    const static uint64_t CRC24CPLUS8 = CRC24CPOLY << 8;
    const static uint32_t k192 = 0x8cfa5500;                 // t=128+64, x^192 mod crc_shifted_poly
    const static uint32_t k128 = 0x6ccc8e00;                 // t=128, x^128 mod crc_shifted_poly
    const static uint32_t k96 = 0x13979900;                  // t=96, x^96 mod crc_shifted_poly
    const static uint32_t k64 = 0x74809300;                  // t=64, x^64 mod crc_shifted_poly
    const static uint64_t u = 0x1c52cdcad;                   // u for crc24C * 256, floor(x^64 / crc_shifted_poly)
    const static uint16_t k_crc_bits = 24;                   // crc_size (in bits)
    const static uint16_t k_crc_bytes = (k_crc_bits-1)/8+1;  // crc size (bytes)
    const static uint32_t k_init_value = 0xffffff00;  // initialisation value (24 1's)
};

__m128i fold_stage_sse(__m128i* dataIn, uint32_t len, struct crc24c_1_parameter* crc24c_1_p)
{
    bool IS_ALIGNED = 1;
    if (len%8)
        IS_ALIGNED = 0;

    const __m128i iota = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    // 1) fold by 128bit. remaining length <=2*128bits
    // set constants and load 1st 16 bytes (128bits) of data, do endian swap
    const __m128i k_set1 = _mm_set_epi32(0, crc24c_1_p->k192, 0, crc24c_1_p->k128);

    // Load 1st 128bits of data and endian swap. Preserve for later use
    __m128i foldData = _mm_shuffle_epi8(dataIn[0], k_endian_shuf_mask128);
    __m128i previous_original = foldData;

    // Handle initialising data with 1's at start of data sequence
    // Append CRC length number of 1's to the start of the data sequence for CRCs required
    // to support this, such as CRC24C_1.
    if (crc24c_1_p->k_init_value)
    {
        // shift data to right by CRC size and append initialised value (usually 1's)
        foldData = bit_shift_right_m128i_sse(foldData, crc24c_1_p->k_crc_bits);
        __m128i k_init_mask = _mm_set_epi32(crc24c_1_p->k_init_value,0,0,0);
        // auto foldData_temp = _mm_maskz_or_epi32 (0xff, foldData, k_init_mask);
        // SSE: _mm_maskz_or_epi32, since _mm_maskz_or_epi32 mask = 0x0F, all or 32bits is used
        __m128i foldData_temp = _mm_or_si128 (foldData, k_init_mask);
        foldData = foldData_temp;
        // Update the length to include appended value
        len = len + crc24c_1_p->k_crc_bits;
    }

    // preserve state of this data
    auto previous_data = foldData;
    auto previous_init = foldData;

    // Determine length of data (whole bytes)
    const int len_bytes = ((len-1)/8)+1;

    // Determine pad size and if non-aligned, pad data to re-align
    int pad_size = 0;
    if (!IS_ALIGNED)
    {
        pad_size = len_bytes*8 - len;
        foldData = bit_shift_right_m128i_sse(foldData, pad_size);
    }

    // If len is 32bytes (256bits) or more, process & fold 16 byte sections at a time
    for (int i=1; i < (int)(len_bytes/16); i++)
    {
        previous_data = previous_original;

        // Load next 16 bytes of data & endian swap
        auto nextData_stage1 = _mm_shuffle_epi8(dataIn[i], k_endian_shuf_mask128);
        previous_original = nextData_stage1;

        // Handle initialising data with 1's at start of data sequence
        if (crc24c_1_p->k_init_value)
        {
            nextData_stage1 = shift_right_256_sse(previous_data, nextData_stage1, crc24c_1_p->k_crc_bits);
        }
        auto previous_data_reserved = nextData_stage1;

        // For none byte aligned data, need to preserve bits that are shifted off the end of
        // the m128i register when padding applied. Isolate these bits and append to the
        // start of the next byte of data.
        if (!IS_ALIGNED)
        {
            nextData_stage1 = shift_right_256_sse(previous_init, nextData_stage1, pad_size);
        }

        // Fold data using clmul & xor with next data
        const auto fold128_clmul192_value = _mm_clmulepi64_si128(foldData, k_set1, 0x00);
        const auto fold128_clmul128_value = _mm_clmulepi64_si128(foldData, k_set1, 0x11);
        // foldData = _mm_ternarylogic_epi32(fold128_clmul192_value, fold128_clmul128_value, nextData_stage1, 0x96);
        __m128i t0 = _mm_xor_si128(fold128_clmul128_value, nextData_stage1);
        foldData = _mm_xor_si128(fold128_clmul192_value, t0);
        previous_init = previous_data_reserved;
    }

    // 2) If remaining length > 128 bits, then pad zero to the most significant bit to align with
    // 256bits, then fold once to 128 bits.
    __m128i fold128Result = foldData;

    // When greater than 16 bytes, load remaining len%16 bytes and maybe some garbage bytes.
    if (len_bytes > 16)
    {
        // load next 16 bytes of data & endian swap
        auto data_stage2 = dataIn[len_bytes/16];
        auto nextData_stage2 = _mm_shuffle_epi8(data_stage2, k_endian_shuf_mask128);

        // Handle initialising data with 1's at start of data sequence
        if (crc24c_1_p->k_init_value)
        {
            nextData_stage2 = shift_right_256_sse(previous_original, nextData_stage2, crc24c_1_p->k_crc_bits);
        }

        // For none byte aligned data, need to preserve bits that are shifted off the end of
        // the m128i register when padding applied. Isolate these bits and append to the
        // start of the next byte of data.
        if (!IS_ALIGNED)
        {
            auto previous_loaded_data = _mm_shuffle_epi8(dataIn[(len_bytes/16)-1], k_endian_shuf_mask128);

            // Handle initialising data with 1's at start of data sequence
            if (crc24c_1_p->k_init_value)
            {
                previous_loaded_data = previous_init;
            }

            nextData_stage2 = shift_right_256_sse(previous_loaded_data, nextData_stage2, pad_size);
        }

        // Determine byte shifts required for padding operations
        const auto byte_shift = (char)(16 - len_bytes % 16);

        // Setup shift value, rotate foldData, so top block in lsb position & bottom black in msb position
        // Note: bit position corresponds to the destination bit position
        //       and value corresponds to the source bit position
        const auto shift_value = _mm_add_epi8(iota, _mm_set1_epi8(byte_shift)); // all i8 in iota plus 13(D)
        __m128i shifted_foldData = _mm_shuffle_epi8(foldData, shift_value);

        // Rotate nextData
        const auto shifted_nextData = _mm_shuffle_epi8(nextData_stage2, shift_value); // original endian swapped low 24bits

        // Move top block from foldData into nextData
        __mmask16 next_mask = (__mmask16)(0xFFFF << (16-byte_shift));
        //const auto new_nextData_stage2 = _mm_mask_shuffle_epi8(shifted_nextData, next_mask, shifted_foldData, iota);
        __m128i next_mask_v;
        mask_make_128i(next_mask, (uint8_t*)(&next_mask_v));
        __m128i new_nextData_stage2 = _mm_blendv_epi8(shifted_nextData, _mm_shuffle_epi8(shifted_foldData, iota), next_mask_v);

        // Clear remaining bits in foldData
        __mmask16 fold_mask = (__mmask16)~(next_mask);
        // foldData = _mm_maskz_mov_epi8(fold_mask, shifted_foldData);
        __m128i fold_mask_v;        
        mask_make_128i(fold_mask, (uint8_t*)(&fold_mask_v));
        foldData = _mm_blendv_epi8(_mm_set1_epi64x(0), shifted_foldData, fold_mask_v);

        // Fold the padded data
        const auto pad128_clmul192_value = _mm_clmulepi64_si128(foldData, k_set1, 0x00); //k_set1:k_192.k_128
        const auto pad128_clmul128_value = _mm_clmulepi64_si128(foldData, k_set1, 0x11);
        // Use ternary logic to perform 2 xor operations (spec by 0x96) on the above clmul results
        // fold128Result = _mm_ternarylogic_epi32(pad128_clmul192_value, pad128_clmul128_value, new_nextData_stage2, 0x96);
        __m128i t1 = _mm_xor_si128(pad128_clmul128_value, new_nextData_stage2);
        fold128Result  = _mm_xor_si128(pad128_clmul192_value, t1);
    }
    else
    {
        // Less than 16 bytes, so pad out with zeros
        const auto num_bytes = (char)(16 - len_bytes);
        const auto shift_value = _mm_add_epi8(iota, _mm_set1_epi8(num_bytes));
        const auto mask = (__mmask16)(0xFFFF >> num_bytes);
        __m128i mask_v; 
        mask_make_128i(mask, (uint8_t*)(&mask_v));
        fold128Result = _mm_blendv_epi8(_mm_set1_epi64x(0), _mm_shuffle_epi8(foldData, shift_value), mask_v);
    }

    // 3) apply 64 bits fold to 64 bits + 32 bits crc(32 bits zero)
    const auto k_set2 = _mm_set_epi64x(crc24c_1_p->k64, crc24c_1_p->k96);
    const auto fold64_clmul96_value = _mm_clmulepi64_si128(fold128Result, k_set2, 0x01);

    // realign data to center, packed with 0's
    auto fold64_realign = _mm_slli_si128(fold128Result, 8);
    fold64_realign = _mm_srli_si128(fold64_realign, 4);
    const auto fold64Result = _mm_xor_si128(fold64_clmul96_value, fold64_realign);

    // 4) Apply 32 bits fold to 32 bits + 32 bits crc(32 bits zero)
    const auto fold32_clmul64_value = _mm_clmulepi64_si128(fold64Result, k_set2, 0x11);
    auto fold32_realign = _mm_slli_si128(fold64Result, 8);
    fold32_realign = _mm_srli_si128(fold32_realign, 8);
    __m128i fold_4stage_result = _mm_xor_si128(fold32_clmul64_value, fold32_realign);

    return fold_4stage_result;

}

__m128i barrett_reduction_sse(__m128i fold_4stage_result, struct crc24c_1_parameter* crc24c_1_p)
{
    const auto k_br_constants = _mm_set_epi32(1, (crc24c_1_p->CRC24CPLUS8 & 0xFFFFFFFF), 1, (crc24c_1_p->u & 0xFFFFFFFF)); 
    const auto br_realign = _mm_srli_si128(fold_4stage_result, 4); 
    const auto clmul_u_value = _mm_clmulepi64_si128(k_br_constants, br_realign, 0x00);
    const auto br_clmul_realign = _mm_srli_si128(clmul_u_value, 4);
    const auto br_clmul = _mm_clmulepi64_si128(k_br_constants, br_clmul_realign, 0x01);
    //return (_mm_maskz_xor_epi32(0x01, fold_data, br_clmul));
    __m128i xmm0 = _mm_xor_si128(fold_4stage_result, br_clmul);
    __m128i xmm1 = _mm_set_epi32(0, 0, 0, 0xFFFFFFFF);
    __m128i br_result = _mm_and_si128 (xmm0, xmm1);
    return br_result;
}

void bblib_lte_crc24c_1_gen_sse(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    __m128i* dataIn = (__m128i*)request->data;
    uint8_t *dataOut = response->data;

    // Determine data length in whole bytes, based on passed in length in bits.
    const int data_byte_len = ((request->len-1)/8)+1;
    struct crc24c_1_parameter crc24c_1;

    __m128i fold_4stage_result = fold_stage_sse(dataIn, request->len, &crc24c_1);
    __m128i br_result = barrett_reduction_sse(fold_4stage_result, &crc24c_1);

    // 2) Write CRC value (br_result) into response structure
    // Align CRC to right of 32bit segment, then further align CRC value based on CRC size
    constexpr uint32_t crc_shift_value = (32-crc24c_1.k_crc_bytes*8)+(crc24c_1.k_crc_bytes*8-crc24c_1.k_crc_bits);
    const auto align_br_result = _mm_srli_epi32(br_result, crc_shift_value);
    // _mm_mask_storeu_epi8(&response->crc_value, 0x000f, align_br_result); // store 0~31bits
    _mm_storeu_si32(&response->crc_value, align_br_result);

    // 3) Append CRC to end of Data steam
    // Identify index to last byte (part or whole)
    const int end_data_idx = data_byte_len - 1;

    // Determine padding size
    uint32_t pad_size = (data_byte_len*8 - request->len) + 24;
    //const auto pad_size_r = _mm_maskz_set1_epi32(0x01, pad_size);
    __m128i pad_size_r = _mm_set_epi32(0, 0, 0, pad_size);

    // Load last byte of data (endian swapped), using mask to ensure rest of data is zeroed
    // then pad data to RHS of 32bit word ready to be merged with CRC
    //const auto end_data_es = _mm_maskz_shuffle_epi8(0x8000, *(__m128i*)(request->data+end_data_idx), k_endian_shuf_mask128);
    __m128i end_data_mask_v = _mm_set_epi64x(0x8000000000000000, 0);
    __m128i end_data_es = _mm_blendv_epi8(_mm_set1_epi64x(0), _mm_shuffle_epi8(*(__m128i*)(request->data+end_data_idx), k_endian_shuf_mask128), end_data_mask_v);
    __m128i pad_end_data = _mm_srl_epi32(end_data_es, pad_size_r);

    // Merge data with CRC value (br_result) and shift back
    // const auto merged_crc = _mm_mask_shuffle_epi8(pad_end_data, 0x0f00, br_result, k_shuf_mask);
    __m128i merged_crc_mask_v = _mm_set_epi64x(0x0000000080808080, 0);
    __m128i merged_crc = _mm_blendv_epi8(pad_end_data, _mm_shuffle_epi8(br_result, k_shuf_mask), merged_crc_mask_v);
    const auto appended_data = _mm_sll_epi64(merged_crc, pad_size_r);

    // Write CRC appended data to memory (needs an endian swap)
    const auto appended_data_es = _mm_shuffle_epi8(appended_data, k_endian_shuf_mask128);
    // _mm_mask_storeu_epi8 (dataOut+end_data_idx, 0xffff, appended_data_es);
    _mm_storeu_si128((__m128i*)(dataOut+end_data_idx), appended_data_es); // store 128bits

    // Update length of CRC appended data
    response->len = request->len + crc24c_1.k_crc_bits;
}

void bblib_lte_crc24c_1_check_sse(bblib_crc_request *request, bblib_crc_response *response)
{
    const static uint16_t k_crc_bits = 24;                   // crc_size (in bits)
    const static uint16_t k_crc_bytes = (k_crc_bits-1)/8+1;  // crc size (bytes)

    // Identify load point in data, which includes start of CRC bits
    const int load_idx = request->len/8;

    // determine alignment shift for left alignment of crc
    const int align_shift = request->len%8;
    const auto align_shift_r = _mm_setr_epi32(align_shift,0,0,0);

    // Extract CRC which may include some bits from the end of data
    // Load last bytes of data + CRC (endian swapped), masking out rest of data.
    // const auto crc_end_data = _mm_maskz_shuffle_epi8 (0x000f, *(__m128i*)(request->data+load_idx), k_endian_shuf_mask32);
    __m128i crc_end_data_mask_v = _mm_set_epi64x(0, 0x0000000080808080);
    __m128i crc_end_data = _mm_blendv_epi8(_mm_set1_epi64x(0), _mm_shuffle_epi8(*(__m128i*)(request->data+load_idx), k_endian_shuf_mask32), crc_end_data_mask_v);

    // Bit left align CRC to remove remaining bits of data
    const auto crc_no_data = _mm_sll_epi32(crc_end_data, align_shift_r);

    // Byte align to right of 32bit word, taking into account size of CRC (6, 11 etc)
    static constexpr uint16_t k_crc_alignment_shift = (4-k_crc_bytes)*8 + (k_crc_bytes*8 - k_crc_bits);
    const auto crc = _mm_srli_epi32(crc_no_data, k_crc_alignment_shift);
    uint32_t *p_crc = (uint32_t*)&crc;

    // Generate CRC & compare results with crc from data (orig_crc)
    bblib_lte_crc24c_1_gen_sse(request, response);
    response->check_passed = (response->crc_value == *p_crc);
}


void bblib_lte_crc16_gen_sse(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    uint8_t *data = request->data;
    uint8_t *dataOut = response->data;

    // len is passed in as bits, so turn into bytes
    uint32_t len_bytes = request->len / 8;

    /* record the start of input data */
    int32_t    i = 0;

    /* A= B mod C => A*K = B*K mod C*K, set K = 2^8, then compute CRC-32, the most significant
     * 16 bits is final 16 bit CRC. */
    const static uint64_t  CRC16POLY = 0x11021;    //CRC16 Polynomial
    const static uint64_t  CRC16PLUS16 = CRC16POLY << 16;  //pads poly to 32bits

    /* some pre-computed key constants */
    const static uint32_t k192   = 0xd5f60000;   //t=128+64, x^192 mod CRC16PLUS16, verified
    const static uint32_t k128   = 0x45630000;   //t=128, x^128 mod CRC16PLUS16, verified
    const static uint32_t k96    = 0xeb230000;   //t=96, x^96 mod CRC16PLUS16, verified
    const static uint32_t k64    = 0xaa510000;   //t=64, x^64 mod CRC16PLUS16, verified
    const static uint64_t u      = 0x111303471;  //u for crc16 * 256, floor(x^64 / CRC16PULS16), verified
    const static __m128i ENDIA_SHUF_MASK = _mm_set_epi8(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                                                  0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
    /* variables */
    __m128i xmm3, xmm2, xmm1, xmm0;

    /* 1. fold by 128bit. remaining length <=2*128bits. */
    xmm3 = _mm_set_epi32(0, k192, 0, k128);
    xmm1 = _mm_load_si128((__m128i *)data); data += 16;
    xmm1 = _mm_shuffle_epi8(xmm1, ENDIA_SHUF_MASK);

    for (i=(len_bytes>>4)-1; i>0; i--){
        xmm2 = xmm1;
        xmm0 = _mm_load_si128((__m128i *)data);  data += 16;
        xmm0 = _mm_shuffle_epi8(xmm0, ENDIA_SHUF_MASK);
        xmm1 = _mm_clmulepi64_si128(xmm1, xmm3, 0x00);
        xmm2 = _mm_clmulepi64_si128(xmm2, xmm3, 0x11);
        xmm0 = _mm_xor_si128(xmm1, xmm0);
        xmm1 = _mm_xor_si128(xmm2, xmm0);
    }

    /* 2. if remaining length > 128 bits, then pad zero to the most-significant bit to grow to 256bits length,
         *   then fold once to 128 bits. */
    if (16 < len_bytes){
        xmm0 = _mm_load_si128((__m128i *)data); //load remaining len%16 bytes and maybe some garbage bytes.
        xmm0 = _mm_shuffle_epi8(xmm0, ENDIA_SHUF_MASK);
        for (i=15-len_bytes%16; i>=0; i--){
            xmm0 = _mm_srli_si128(xmm0, 1);
            xmm0 = _mm_insert_epi8(xmm0, _mm_extract_epi8(xmm1, 0), 15);
            xmm1 = _mm_srli_si128(xmm1, 1);
        }
        xmm2 = xmm1;
        xmm1 = _mm_clmulepi64_si128(xmm1, xmm3, 0x00);
        xmm2 = _mm_clmulepi64_si128(xmm2, xmm3, 0x11);
        xmm0 = _mm_xor_si128(xmm1, xmm0);
        xmm0 = _mm_xor_si128(xmm2, xmm0);
    }
    else{
        xmm0 = xmm1;
        for (i=16-len_bytes; i>0; i--){
            xmm0 = _mm_srli_si128(xmm0, 1);
        }
    }

    /* 3. Apply 64 bits fold to 64 bits + 32 bits crc(32 bits zero) */
    xmm3 =  _mm_set_epi32(0, 0, 0, k96);
    xmm1 = _mm_clmulepi64_si128(xmm0, xmm3, 0x01);
    xmm2 = _mm_slli_si128(xmm0, 8);
    xmm2 = _mm_srli_si128(xmm2, 4);
    xmm0 = _mm_xor_si128(xmm1, xmm2);

    /* 4. Apply 32 bits fold to 32 bits + 32 bits crc(32 bits zero) */
    xmm3 =  _mm_set_epi32(0, 0, 0, k64);
    xmm1 = _mm_clmulepi64_si128(xmm0, xmm3, 0x01);
    xmm0 = _mm_slli_si128(xmm0, 8);
    xmm0 = _mm_srli_si128(xmm0, 8);
    xmm0 = _mm_xor_si128(xmm1, xmm0);


    /* 5. Use Barrett Reduction Algorithm to calculate the 32 bits crc.
     * Output: C(x)  = R(x) mod P(x)
     * Step 1: T1(x) = floor(R(x)/x^32)) * u
     * Step 2: T2(x) = floor(T1(x)/x^32)) * P(x)
     * Step 3: C(x)  = R(x) xor T2(x) mod x^32 */
    xmm1 = _mm_set_epi32(0, 0, 1, (u & 0xFFFFFFFF));
    xmm2 = _mm_srli_si128(xmm0, 4);
    xmm1 = _mm_clmulepi64_si128(xmm1, xmm2, 0x00);
    xmm1 = _mm_srli_si128(xmm1, 4);
    xmm2 =  _mm_set_epi32(0, 0, 1, (CRC16PLUS16 & 0xFFFFFFFF));
    xmm1 = _mm_clmulepi64_si128(xmm1, xmm2, 0x00);
    xmm0 = _mm_xor_si128(xmm0, xmm1);
    xmm1 = _mm_set_epi32(0, 0, 0, 0xFFFFFFFF);
    xmm0 = _mm_and_si128 (xmm0, xmm1);


    /* 6. Update Result */
    /* add crc to last 2 bytes. */
    dataOut[len_bytes]   =  _mm_extract_epi8(xmm0, 3);
    dataOut[len_bytes+1] =  _mm_extract_epi8(xmm0, 2);
    response->len = (len_bytes + 2)*8;

    /* the most significant 24 bits of the 32 bits crc is the finial 24 bits crc. */
    response->crc_value = (((uint32_t)_mm_extract_epi32(xmm0, 0)) >> 16);

}



void bblib_lte_crc16_check_sse(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    if (request->data == NULL){
        printf("bblib_lte_crc16_check input / output address error \n");
        response->check_passed = false;
        return;
    }

    // len is passed in as bits, so turn into bytes
    uint32_t len_bytes = request->len / 8;

    /* CRC in the original sequence */
    uint32_t CRC_orig = 0;

    CRC_orig = ((request->data[len_bytes]<<8)&0x0000FF00) +
               (request->data[len_bytes+1]&0x000000FF);
    bblib_lte_crc16_gen_sse(request, response);

    if (response->crc_value != CRC_orig)
        response->check_passed = false;
    else
        response->check_passed = true;
}



void bblib_lte_crc11_gen_sse(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    _mm256_zeroupper();
    uint8_t *data = request->data;
    uint8_t *dataOut = response->data;

    // len is passed in as bits, so turn to bytes
    uint32_t len_bytes = request->len / 8;

    /* record the start of input data */
    int32_t    i = 0;

    /* A= B mod C => A*K = B*K mod C*K, set K = 2^8, then compute CRC-32, the most significant
     * 11 bits is final 11 bit CRC. */
    const static uint64_t  CRC11POLY = 0xe21;    //CRC11 Polynomial
    const static uint64_t  CRC11PLUS21 = CRC11POLY << 21;  //pads poly to 32bits

    /* some pre-computed key constants */
    const static uint32_t k192   = 0x8ea00000;   //t=128+64, x^192 mod CRC11PLUS21, verified
    const static uint32_t k128   = 0x47600000;   //t=128, x^128 mod CRC11PLUS21, verified
    const static uint32_t k96    = 0x5e600000;   //t=96, x^96 mod CRC11PLUS21, verified
    const static uint32_t k64    = 0xc9000000;   //t=64, x^64 mod CRC11PLUS21, verified
    const static uint64_t u      = 0x1b3fa1f48;  //u for crc11 * 256, floor(x^64 / CRC11PLUS21), verified
    const static __m128i ENDIA_SHUF_MASK = _mm_set_epi8(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                                                  0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);

    /* variables */
    __m128i xmm3, xmm2, xmm1, xmm0;

    /* 1. fold by 128bit. remaining length <=2*128bits. */
    xmm3 = _mm_set_epi32(0, k192, 0, k128);
    xmm1 = _mm_load_si128((__m128i *)data); data += 16;
    xmm1 = _mm_shuffle_epi8(xmm1, ENDIA_SHUF_MASK);

    for (i=(len_bytes>>4)-1; i>0; i--){
        xmm2 = xmm1;
        xmm0 = _mm_load_si128((__m128i *)data);  data += 16;
        xmm0 = _mm_shuffle_epi8(xmm0, ENDIA_SHUF_MASK);
        xmm1 = _mm_clmulepi64_si128(xmm1, xmm3, 0x00);
        xmm2 = _mm_clmulepi64_si128(xmm2, xmm3, 0x11);
        xmm0 = _mm_xor_si128(xmm1, xmm0);
        xmm1 = _mm_xor_si128(xmm2, xmm0);
    }

    /* 2. if remaining length > 128 bits, then pad zero to the most-significant bit to grow to 256bits length,
        *   then fold once to 128 bits. */
    if (16 < len_bytes){
        xmm0 = _mm_load_si128((__m128i *)data); //load remaining len%16 bytes and maybe some garbage bytes.
        xmm0 = _mm_shuffle_epi8(xmm0, ENDIA_SHUF_MASK);
        for (i=15-len_bytes%16; i>=0; i--){
            xmm0 = _mm_srli_si128(xmm0, 1);
            xmm0 = _mm_insert_epi8(xmm0, _mm_extract_epi8(xmm1, 0), 15);
            xmm1 = _mm_srli_si128(xmm1, 1);
        }
        xmm2 = xmm1;
        xmm1 = _mm_clmulepi64_si128(xmm1, xmm3, 0x00);
        xmm2 = _mm_clmulepi64_si128(xmm2, xmm3, 0x11);
        xmm0 = _mm_xor_si128(xmm1, xmm0);
        xmm0 = _mm_xor_si128(xmm2, xmm0);
    }
    else{
        xmm0 = xmm1;
        for (i=16-len_bytes; i>0; i--){
            xmm0 = _mm_srli_si128(xmm0, 1);
        }
    }

    /* 3. Apply 64 bits fold to 64 bits + 32 bits crc(32 bits zero) */
    xmm3 =  _mm_set_epi32(0, 0, 0, k96);
    xmm1 = _mm_clmulepi64_si128(xmm0, xmm3, 0x01);
    xmm2 = _mm_slli_si128(xmm0, 8);
    xmm2 = _mm_srli_si128(xmm2, 4);
    xmm0 = _mm_xor_si128(xmm1, xmm2);

    /* 4. Apply 32 bits fold to 32 bits + 32 bits crc(32 bits zero) */
    xmm3 =  _mm_set_epi32(0, 0, 0, k64);
    xmm1 = _mm_clmulepi64_si128(xmm0, xmm3, 0x01);
    xmm0 = _mm_slli_si128(xmm0, 8);
    xmm0 = _mm_srli_si128(xmm0, 8);
    xmm0 = _mm_xor_si128(xmm1, xmm0);


    /* 5. Use Barrett Reduction Algorithm to calculate the 32 bits crc.
     * Output: C(x)  = R(x) mod P(x)
     * Step 1: T1(x) = floor(R(x)/x^32)) * u
     * Step 2: T2(x) = floor(T1(x)/x^32)) * P(x)
     * Step 3: C(x)  = R(x) xor T2(x) mod x^32 */
    xmm1 = _mm_set_epi32(0, 0, 1, (u & 0xFFFFFFFF));
    xmm2 = _mm_srli_si128(xmm0, 4);
    xmm1 = _mm_clmulepi64_si128(xmm1, xmm2, 0x00);
    xmm1 = _mm_srli_si128(xmm1, 4);
    xmm2 =  _mm_set_epi32(0, 0, 1, (CRC11PLUS21 & 0xFFFFFFFF));
    xmm1 = _mm_clmulepi64_si128(xmm1, xmm2, 0x00);
    xmm0 = _mm_xor_si128(xmm0, xmm1);
    xmm1 = _mm_set_epi32(0, 0, 0, 0xFFFFFFFF);
    xmm0 = _mm_and_si128 (xmm0, xmm1);


    /* 6. Update Result */
    /* add crc to last byte. */
    dataOut[len_bytes]   =  _mm_extract_epi8(xmm0, 3);
    dataOut[len_bytes+1] =  _mm_extract_epi8(xmm0, 2);
    response->len = ((len_bytes + 2)*8) - 5;

    /* the most significant 11 bits of the 32 bits crc is the final 11 bits crc. */
    response->crc_value = (((uint32_t)_mm_extract_epi32(xmm0, 0)) >> 21);
    _mm256_zeroupper();
}



void bblib_lte_crc11_check_sse(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    if (request->data == NULL) {
        printf("bblib_lte_crc11_check input / output address error \n");
        response->check_passed = false;
        return;
    }
    // len is passed in as bits, so turn into bytes
    uint32_t len_bytes = request->len / 8;

    /* CRC in the original sequence */
    uint32_t CRC_orig = 0;

    CRC_orig = (((request->data[len_bytes]<<8)&0x0000FF00) +
               ((request->data[len_bytes+1])&0x000000FF)) >> 5;
    bblib_lte_crc11_gen_sse(request, response);

    if (response->crc_value != CRC_orig)
        response->check_passed = false;
    else
        response->check_passed = true;
}



void bblib_lte_crc6_gen_sse(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    _mm256_zeroupper();
    uint8_t *data = request->data;
    uint8_t *dataOut = response->data;

    // len is passed in as bits, so turn to bytes
    uint32_t len_bytes = request->len / 8;

    /* record the start of input data */
    int32_t    i = 0;

    /* A= B mod C => A*K = B*K mod C*K, set K = 2^8, then compute CRC-32, the most significant
     * 6 bits is final 6 bit CRC. */
    const static uint64_t  CRC6POLY = 0x61;    //CRC6 Polynomial
    const static uint64_t  CRC6PLUS26 = CRC6POLY << 26;  //pads poly to 32bits

    /* some pre-computed key constants */
    const static uint32_t k192   = 0x38000000;   //t=128+64, x^192 mod CRC6PLUS26, verified
    const static uint32_t k128   = 0x1c000000;   //t=128, x^128 mod CRC6PLUS26, verified
    const static uint32_t k96    = 0x8c000000;   //t=96, x^96 mod CRC6PLUS26, verified
    const static uint32_t k64    = 0xcc000000;   //t=64, x^64 mod CRC6PLUS26, verified
    const static uint64_t u      = 0x1fab37693;  //u for crc11 * 256, floor(x^64 / CRC6PLUS26), verified
    const static __m128i ENDIA_SHUF_MASK = _mm_set_epi8(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                                                  0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);

    /* variables */
    __m128i xmm3, xmm2, xmm1, xmm0;

    /* 1. fold by 128bit. remaining length <=2*128bits. */
    xmm3 = _mm_set_epi32(0, k192, 0, k128);
    xmm1 = _mm_load_si128((__m128i *)data); data += 16;
    xmm1 = _mm_shuffle_epi8(xmm1, ENDIA_SHUF_MASK);

    for (i=(len_bytes>>4)-1; i>0; i--){
        xmm2 = xmm1;
        xmm0 = _mm_load_si128((__m128i *)data);  data += 16;
        xmm0 = _mm_shuffle_epi8(xmm0, ENDIA_SHUF_MASK);
        xmm1 = _mm_clmulepi64_si128(xmm1, xmm3, 0x00);
        xmm2 = _mm_clmulepi64_si128(xmm2, xmm3, 0x11);
        xmm0 = _mm_xor_si128(xmm1, xmm0);
        xmm1 = _mm_xor_si128(xmm2, xmm0);
    }

    /* 2. if remaining length > 128 bits, then pad zero to the most-significant bit to grow to 256bits length,
         *   then fold once to 128 bits. */
    if (16 < len_bytes){
        xmm0 = _mm_load_si128((__m128i *)data); //load remaining len%16 bytes and maybe some garbage bytes.
        xmm0 = _mm_shuffle_epi8(xmm0, ENDIA_SHUF_MASK);
        for (i=15-len_bytes%16; i>=0; i--){
            xmm0 = _mm_srli_si128(xmm0, 1);
            xmm0 = _mm_insert_epi8(xmm0, _mm_extract_epi8(xmm1, 0), 15);
            xmm1 = _mm_srli_si128(xmm1, 1);
        }
        xmm2 = xmm1;
        xmm1 = _mm_clmulepi64_si128(xmm1, xmm3, 0x00);
        xmm2 = _mm_clmulepi64_si128(xmm2, xmm3, 0x11);
        xmm0 = _mm_xor_si128(xmm1, xmm0);
        xmm0 = _mm_xor_si128(xmm2, xmm0);
    }
    else{
        xmm0 = xmm1;
        for (i=16-len_bytes; i>0; i--){
            xmm0 = _mm_srli_si128(xmm0, 1);
        }
    }

    /* 3. Apply 64 bits fold to 64 bits + 32 bits crc(32 bits zero) */
    xmm3 =  _mm_set_epi32(0, 0, 0, k96);
    xmm1 = _mm_clmulepi64_si128(xmm0, xmm3, 0x01);
    xmm2 = _mm_slli_si128(xmm0, 8);
    xmm2 = _mm_srli_si128(xmm2, 4);
    xmm0 = _mm_xor_si128(xmm1, xmm2);

    /* 4. Apply 32 bits fold to 32 bits + 32 bits crc(32 bits zero) */
    xmm3 =  _mm_set_epi32(0, 0, 0, k64);
    xmm1 = _mm_clmulepi64_si128(xmm0, xmm3, 0x01);
    xmm0 = _mm_slli_si128(xmm0, 8);
    xmm0 = _mm_srli_si128(xmm0, 8);
    xmm0 = _mm_xor_si128(xmm1, xmm0);


    /* 5. Use Barrett Reduction Algorithm to calculate the 32 bits crc.
     * Output: C(x)  = R(x) mod P(x)
     * Step 1: T1(x) = floor(R(x)/x^32)) * u
     * Step 2: T2(x) = floor(T1(x)/x^32)) * P(x)
     * Step 3: C(x)  = R(x) xor T2(x) mod x^32 */
    xmm1 = _mm_set_epi32(0, 0, 1, (u & 0xFFFFFFFF));
    xmm2 = _mm_srli_si128(xmm0, 4);
    xmm1 = _mm_clmulepi64_si128(xmm1, xmm2, 0x00);
    xmm1 = _mm_srli_si128(xmm1, 4);
    xmm2 =  _mm_set_epi32(0, 0, 1, (CRC6PLUS26 & 0xFFFFFFFF));
    xmm1 = _mm_clmulepi64_si128(xmm1, xmm2, 0x00);
    xmm0 = _mm_xor_si128(xmm0, xmm1);
    xmm1 = _mm_set_epi32(0, 0, 0, 0xFFFFFFFF);
    xmm0 = _mm_and_si128 (xmm0, xmm1);


    /* 6. Update Result */
    /* add crc to last byte. */
    dataOut[len_bytes]   =  _mm_extract_epi8(xmm0, 3);
    response->len = ((len_bytes + 1)*8) - 2;

    /* the most significant 6 bits of the 32 bits crc is the final 6 bits crc. */
    response->crc_value = (((uint32_t)_mm_extract_epi32(xmm0, 0)) >> 26);
    _mm256_zeroupper();
}



void bblib_lte_crc6_check_sse(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    if (request->data == NULL) {
        printf("bblib_lte_crc6_check input / output address error \n");
        response->check_passed = false;
        return;
    }
    // len is passed in as bits, so turn into bytes
    uint32_t len_bytes = request->len / 8;

    /* CRC in the original sequence */
    uint32_t CRC_orig = 0;

    CRC_orig = ((request->data[len_bytes])&0x000000FF)>>2;
    bblib_lte_crc6_gen_sse(request, response);

    if (response->crc_value != CRC_orig)
        response->check_passed = false;
    else
        response->check_passed = true;
}



#else
void bblib_lte_crc24a_gen_sse(struct bblib_crc_request *request, struct bblib_crc_response *response){
    printf("bblib_crc requires at least SSE4.2 ISA support to run\n");
    exit(-1);
}
void bblib_lte_crc24b_gen_sse(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    printf("bblib_crc requires at least SSE4.2 ISA support to run\n");
    exit(-1);
}
void bblib_lte_crc16_gen_sse(struct bblib_crc_request *request, struct bblib_crc_response *response){
    printf("bblib_crc requires at least SSE4.2 ISA support to run\n");
    exit(-1);
}
void bblib_lte_crc11_gen_sse(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    printf("bblib_crc requires at least SSE4.2 ISA support to run\n");
    exit(-1);
}
void bblib_lte_crc6_gen_sse(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    printf("bblib_crc requires at least SSE4.2 ISA support to run\n");
    exit(-1);
}
void bblib_lte_crc24a_check_sse(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    printf("bblib_crc requires at least SSE4.2 ISA support to run\n");
    exit(-1);
}
void bblib_lte_crc24b_check_sse(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    printf("bblib_crc requires at least SSE4.2 ISA support to run\n");
    exit(-1);
}
void bblib_lte_crc16_check_sse(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    printf("bblib_crc requires at least SSE4.2 ISA support to run\n");
    exit(-1);
}
void bblib_lte_crc11_check_sse(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    printf("bblib_crc requires at least SSE4.2 ISA support to run\n");
    exit(-1);
}
void bblib_lte_crc6_check_sse(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    printf("bblib_crc requires at least SSE4.2 ISA support to run\n");
    exit(-1);
}
#endif
