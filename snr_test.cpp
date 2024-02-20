#include <iostream>
#include <iomanip>      // std::setprecision
#include <vector>
#include <pthread.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sched.h>
#include <utmpx.h>
#include <time.h>
#include <sys/time.h>
#include <signal.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <string.h>
#include <math.h>
#include <immintrin.h> /* AVX */
#include <assert.h>
#include "common_typedef.h"
#include "utility.h"
#include "phy_crc.h"
#include "phy_rate_match.h"
#include "phy_turbo.h"

#define SOFT_VALUE 16
#define SCALE_NOISE 2
#define MAX_TB_SIZE 6120
#define CRC_LEN 24
#define MAX_K MAX_TB_SIZE + CRC_LEN
#define MAX_PKT_NUM 10000
#define MAX_WIN_NUM 4
#define SNR_START   -10
#define SNR_STOP    10
#define SNR_CNT     (SNR_STOP - SNR_STOP + 1)
#define ALIGN_SIZE  512
#define _aligned_malloc(x,y) memalign(y,x)
#define iAssert(p) if(!(p)){fprintf(stderr,\
    "Assertion failed: %s, file %s, line %d\n",\
#p, __FILE__, __LINE__);exit(-1);}


UWORD8 *pEnDataIn = NULL;
UWORD8 *pEnDataInBak = NULL;
UWORD8 *pEnDataOut = NULL;
UWORD8 *pEnOutD0 = NULL;
UWORD8 *pEnOutD1 = NULL;
UWORD8 *pEnOutD2 = NULL;
WORD8 *pModOut = NULL;
WORD8 *pLLR = NULL;
UWORD8 *pDecOut = NULL;
UWORD8 *pTmpBuf1 = NULL;
UWORD8 *pTmpBuf2 = NULL;
UWORD8 *pTmpBuf3 = NULL;
UWORD8 *pTmpBuf4 = NULL;
UWORD8 *pDeRmOut = NULL;
float gSnrCalc[50] = {0.0};


void print_byte(UWORD8 *p, int len, const char *text)
{
    printf("%s", text);
    for(int i=0; i<len; i++)
    {
        printf("%02x ", p[i]);
    }
    printf("\n");
}

void print_bit(UWORD8 *p, int len, const char *text)
{
    printf("%s", text);
    for(int i=0; i<len; i++)
    {
        for(int j=0; j<8; j++)
        {
            printf("%d ", (p[i]>>j)&1);
        }
    }
    printf("\n");
}
void print_bit_big(UWORD8 *p, int len, const char *text)
{
    printf("%s", text);
    for(int i=0; i<len; i++)
    {
        for(int j=7; j>=0; j--)
        {
            printf("%d ", (p[i]>>j)&1);
        }
    }
    printf("\n");
}


pid_t gettid(void)
{
    return syscall(SYS_gettid);
}

int bind_to_cpu(pid_t pid, int cpu)
{
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    return sched_setaffinity(pid, sizeof(mask), &mask);
}

inline UWORD64 GenRand64()
{
    UWORD64 rr;
    WORD32 ret;
    do
    {
        ret = _rdrand64_step(&rr);
    } while (ret == 0);
    return rr;
}

void dataSource(UWORD8 *pData, WORD32 Len)
{
    for(WORD32 i=0;i<Len; i++)
    {
        *pData++ = GenRand64()&0xff;
    }
}

void initialTurbo()
{    
    pEnDataIn = (UWORD8 *)_aligned_malloc(MAX_K/8, ALIGN_SIZE);
    assert(pEnDataIn != NULL);
    pEnDataInBak = (UWORD8 *)_aligned_malloc(MAX_K/8, ALIGN_SIZE);
    assert(pEnDataInBak != NULL);
    pEnDataOut = (UWORD8 *)_aligned_malloc(3*(MAX_K + 4)/8+1,ALIGN_SIZE);
    assert(pEnDataOut != NULL);
    pEnOutD0 = (UWORD8 *)_aligned_malloc((MAX_K + 4)/8+1,ALIGN_SIZE);
    assert(pEnOutD0 != NULL);
    pEnOutD1 = (UWORD8 *)_aligned_malloc((MAX_K + 4)/8+1,ALIGN_SIZE);
    assert(pEnOutD1 != NULL);
    pEnOutD2 = (UWORD8 *)_aligned_malloc((MAX_K + 4)/8+1,ALIGN_SIZE);
    assert(pEnOutD2 != NULL);
    pModOut = (WORD8 *)_aligned_malloc(3*(MAX_K + 4),ALIGN_SIZE);
    assert(pModOut != NULL);
    pLLR = (WORD8 *)_aligned_malloc(3*(MAX_K + 4),ALIGN_SIZE);
    assert(pLLR != NULL);
    pDecOut = (UWORD8 *)_aligned_malloc(MAX_K/8, ALIGN_SIZE);
    assert(pDecOut != NULL);
    pTmpBuf1 = (UWORD8 *)_aligned_malloc(3*(MAX_K + 4),ALIGN_SIZE);
    assert(pTmpBuf1 != NULL);
    pTmpBuf2 = (UWORD8 *)_aligned_malloc(3*(MAX_K + 4),ALIGN_SIZE);
    assert(pTmpBuf2 != NULL);
    pTmpBuf3 = (UWORD8 *)_aligned_malloc(6528 * 16,ALIGN_SIZE);
    assert(pTmpBuf3 != NULL);
    pTmpBuf4 = (UWORD8 *)_aligned_malloc(MAX_K,ALIGN_SIZE);
    assert(pTmpBuf4 != NULL);
    pDeRmOut = (UWORD8 *)_aligned_malloc(3*(MAX_K + 4),ALIGN_SIZE);
    assert(pDeRmOut != NULL);
}

void releaseTurbo()
{
    free(pEnDataIn);
    pEnDataIn = NULL;
    free(pEnDataInBak);
    pEnDataInBak = NULL;
    free(pEnDataOut);
    pEnDataOut = NULL;
    free(pEnOutD0);
    (pEnOutD0) = NULL;
    free(pEnOutD1);
    pEnOutD1 = NULL;
    free(pEnOutD2);
    pEnOutD2 = NULL;
    free(pModOut);
    pModOut = NULL;
    free(pLLR);
    pLLR = NULL;
    //free(pDecOut);
    //pDecOut = NULL;
    free(pTmpBuf1);
    pTmpBuf1 = NULL;
    free(pTmpBuf2);
    pTmpBuf2 = NULL;
    free(pTmpBuf3);
    pTmpBuf3 = NULL;
    free(pTmpBuf4);
    pTmpBuf4 = NULL;
    free(pDeRmOut);
    pDeRmOut = NULL;
}

static uint64_t reverse_in_byte(uint64_t x)
{
    __m512i ymm1;
    __m512i ymm2=_mm512_set_epi8 (8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7,
                              8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7,
                              8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7,
                              8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7);
    uint64_t maskOut;

    ymm1 = _mm512_movm_epi8 (x);
    ymm1 = _mm512_shuffle_epi8(ymm1,ymm2);
    maskOut = _mm512_movepi8_mask (ymm1);
    return maskOut;
}

void modulation(UWORD8 *pIn, WORD8 *pOut, UWORD16 _block_length)
{
    UWORD64 *pInOffset;
    UWORD64 data0;
    __m512i x0;
    WORD8 *pOutOffset;
    
    pInOffset = (UWORD64 *)pIn;
    pOutOffset = pOut;
    for(UWORD16 i=0; i<_block_length; i=i+64)
    {
        data0 = *pInOffset++;
        data0 = reverse_in_byte(data0);//reverse!!!!
        x0 = _mm512_mask_set1_epi8 (_mm512_set1_epi8 (SOFT_VALUE), (__mmask64)data0, -SOFT_VALUE);
        _mm512_storeu_si512(pOutOffset,x0);
        pOutOffset = pOutOffset + 64;
    }
}

/* Adds Gaussian white noise to channel output */
void awgn(UWORD16 codeLen, WORD8 *pModOut, WORD8 *pLLR, float snr, int snridx)
{    
    int  nCol, nIter = 16, idxCol, idx;
    //float snr;
    UWORD64 evenNoise[3000][32]  __attribute__ ((aligned (ALIGN_SIZE)));
    char noise[3000]  __attribute__ ((aligned (ALIGN_SIZE)));
    WORD8 *pOut, *pIn;
    char *input;
    char c1, c2;
    UWORD16 sigma; /* Noise variance */
    __m128i x0, x1, ss;
    __m256i y0, y1, y2, y3, y4, mask;

    //snr = snr + 10.0*log10((float)infoLen/(float)codeLen);
            
    //printf("snr=%f\n",snr);
    nCol = (codeLen >> 4) + 3;
    nCol = nCol & 0xFFFE; // mask last bit to be 0    
    for (idxCol = 0; idxCol < nCol; idxCol ++) 
    {
        for (idx = 0; idx < nIter * 2; idx ++) 
        {
            evenNoise[idxCol][idx] = GenRand64();
        }
    }
    

    ss = _mm_setr_epi16(2, 0, 0, 0, 0, 0, 0, 0);
    mask = _mm256_setr_epi8(0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0,
                        0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0);
    y1 = _mm256_setr_epi16(14188, 14188, 14188, 14188, 14188, 14188, 14188, 14188,
                           14188, 14188, 14188, 14188, 14188, 14188, 14188, 14188);
    
    sigma = floor(sqrt(1.0/1.0)*pow(10, -snr * 0.05) * 8192 + 0.5);
    y2 = _mm256_setr_epi16(sigma, sigma, sigma, sigma, sigma, sigma, sigma, sigma,
                            sigma, sigma, sigma, sigma, sigma, sigma, sigma, sigma);
    y0 = _mm256_mulhrs_epi16(y2, y1);
    input = pModOut; 
    
    for (idxCol = 0; idxCol < nCol; idxCol += 2) 
    {
        y2 = _mm256_setzero_si256();
        for (idx = 0; idx < nIter * 2; idx += 2) 
        {
            x1 = _mm_load_si128((__m128i *)&(evenNoise[idxCol][idx]));
            y1 = _mm256_cvtepi8_epi16(x1);
            y2 = _mm256_add_epi16(y2, y1);
        }
        y2 = _mm256_sra_epi16(y2, ss);
        y2 = _mm256_mulhrs_epi16(y0, y2);

        y4 = _mm256_setzero_si256();
        for (idx = 0; idx < nIter * 2; idx += 2) 
        {
            x1 = _mm_load_si128((__m128i *)&(evenNoise[idxCol + 1][idx]));
            y3 = _mm256_cvtepi8_epi16(x1);
            y4 = _mm256_add_epi16(y4, y3);
        }
        y4 = _mm256_sra_epi16(y4, ss);
        y4 = _mm256_mulhrs_epi16(y0, y4);

        y1 = _mm256_and_si256(y2, mask);
        y1 = _mm256_slli_si256(y1, 1);
        y3 = _mm256_and_si256(y4, mask);
        y1 = _mm256_adds_epi8(y1, y3);

        _mm256_store_si256(( __m256i* )&(noise[idxCol * 16]), y1);
    }
    

    pOut = pLLR; 
    pIn = pModOut;
   

    __int64 noiseE=0;
    __int64 signalE=0;
    float SNR=0;
    for (idxCol = 0; idxCol < codeLen; idxCol ++) 
    {
        c1 = noise[idxCol]*SCALE_NOISE + *(pIn + idxCol);
        noiseE += noise[idxCol]*noise[idxCol]*SCALE_NOISE*SCALE_NOISE;
        signalE += (*(pIn + idxCol)) * (*(pIn + idxCol));
        *(pOut + idxCol) = c1 ;
    }
    //printf("noise=%f\n",noise[idxCol]);
    SNR = 10.0*log10((float)signalE/(float)noiseE);
    //printf("signalE=%lld,noiseE=%lld,SNR=%f,snr=%f\n",signalE,noiseE,SNR,snr);
    gSnrCalc[snridx] += SNR;
    return;
}

/* Generate PER statistics */  
int PERestimation(UWORD16 infoLen, const UWORD8 *pDecOut, const UWORD8 *pEnDataInBak)
{
    int ErrPktsIdx = 0;    
    const UWORD8 *pOrg = pEnDataInBak;
    const UWORD8 *pDec = pDecOut;
     WORD32 sum = 0, sum1 = 0;
    for(int i = 0; i<(infoLen>>3); i++)
    {        
        sum = sum + ((*pOrg++) ^ (*pDec++));
    }
    sum1 = ((*pOrg++)^(*pDec++))&((1<<(infoLen&0x7))-1);
    sum = sum + sum1;
    if(sum != 0)
        return ErrPktsIdx = 1;
    //printf("ErrPktsIdx=%d\n",ErrPktsIdx);
           
    return ErrPktsIdx;
                         
}

void b2s(UWORD8 *pIn, int len, UWORD8 *pOut)
{
    for(int i=0;i<len;i++)
    {
        UWORD8 t = 0;
        
        for(int j=0;j<8;j++)
        {
            t |= ((pIn[i] >> (7 - j)) & 1) << j;
        }

        pOut[i] = t;
    }
}

int bler_snr_test(UWORD16 K, float snrStart, float snrStop, double *pBler)
{
    struct bblib_crc_request crc_request;
    struct bblib_crc_response crc_response;
    struct bblib_rate_match_dl_request rate_match_request;
    struct bblib_rate_match_dl_response rate_match_response;
    struct bblib_turbo_encoder_request turbo_encoder_request;
    struct bblib_turbo_encoder_response turbo_encoder_response;
    struct bblib_rate_match_ul_request de_rm_request;
    struct bblib_rate_match_ul_response de_rm_response;
    struct bblib_turbo_decoder_request dec_request;
    struct bblib_turbo_decoder_response dec_response;
    int nsnrIdx;
    int nPktIdx;          
    int ErrPktsIdx;
    int nErrPkts[MAX_WIN_NUM] = {0};
    double PER[SNR_CNT][MAX_WIN_NUM]={{0}};
    const int nWinSize[MAX_WIN_NUM] = {8,16,32,64};
    int ret;
    pid_t tid;
    const float snrStep = 1.0f;    
    float snr = snrStart - snrStep; // init value
    
    tid = gettid();
    ret = bind_to_cpu(tid, 5);
    iAssert(!ret);

    initialTurbo();

    for(nsnrIdx = 0; nsnrIdx < SNR_CNT; nsnrIdx++)
    {
        snr = snr + snrStep;
        memset(nErrPkts, 0, sizeof(nErrPkts));
        printf("snr %f\n",snr);
        for(nPktIdx = 0; nPktIdx < MAX_PKT_NUM; nPktIdx++)
        {
            /* 1. Packet generation */
            dataSource(pEnDataIn,(K-CRC_LEN)>>3);

            
            memcpy(pEnDataInBak, pEnDataIn, (K-CRC_LEN)>>3);

            /* 2. TB CRC */
            crc_request.data = pEnDataIn;
            crc_request.len = K-CRC_LEN;
            crc_response.data = pEnDataIn;
            bblib_lte_crc24a_gen_avx512(&crc_request, &crc_response);

            //print_byte(pEnDataIn, (K)/8, "  CRC OUT: ");

            /* 3. encoder */
            turbo_encoder_request.length = K>>3;
            turbo_encoder_request.case_id = K/8-4;
            turbo_encoder_request.input_win = pEnDataIn;
            turbo_encoder_response.output_win_0 = pEnOutD0;
            turbo_encoder_response.output_win_1 = pEnOutD1;
            turbo_encoder_response.output_win_2 = pEnOutD2;
            bblib_turbo_encoder(&turbo_encoder_request, &turbo_encoder_response);

            //print_byte(pEnOutD0, K/8+1, " DEC OUT0: ");
            //print_byte(pEnOutD1, K/8+1, " DEC OUT1: ");
            //print_byte(pEnOutD2, K/8+1, " DEC OUT2: ");
            //print_bit(pEnOutD0, K/8+1, " DEC OUT0: ");
            
            /* 4. rate matching */
            rate_match_request.C = 1;                                  // Total number of code blocks
            rate_match_request.direction = 0;                                // flag of DL or UL, 1 for DL and 0 for UL
            rate_match_request.G = (K + 4) * 3;                            // length of bits before modulation for 1 UE in 1 subframe
            rate_match_request.NL = 1;                          // Number of layer
            rate_match_request.Qm = 2;//1;                               // Modulation type, which can be 2/4/6
            rate_match_request.rvidx = 0;                          // Redundancy version, which can be 0/1/2/3
            rate_match_request.bypass_rvidx = 1;
            rate_match_request.Kidx = K/8-5;                      // Position in turbo code internal interleave table, Kidx=i-1 in TS 136.212 table 5.1.3-3
            rate_match_request.r = 0;                              // index of current code block in all code blocks
            rate_match_request.nLen = K + 4;     // Length of input data from tin0/tin1/tin2 in bits,  nLen=K(Kidx+1)+4 in TS 136.212 table 5.1.3-3
            rate_match_request.tin0 = pEnOutD0;       // pointer to input stream 0 from turbo encoder
            rate_match_request.tin1 = pEnOutD1;       // tin1 pointer to input stream 1 from turbo encoder
            rate_match_request.tin2 = pEnOutD2;       // tin2 pointer to input stream 2 from turbo encoder
            rate_match_response.output = pEnDataOut; // output buffer for data stream after rate matching
            bblib_rate_match_dl(&rate_match_request, &rate_match_response);

            //print_byte(pEnDataOut, (rate_match_request.G + 7) / 8, "   RM OUT: ");
            //print_bit_big(pEnDataOut, (rate_match_request.G + 7) / 8, "   RM OUT: ");

            /* 6. Modulation */
            modulation(pEnDataOut, pModOut, rate_match_request.G);
            
            //print_byte((UWORD8 *)pModOut, rate_match_request.G , "  MOD OUT: ");

            /* 7. Add Noise */
            awgn(rate_match_request.G, pModOut, pLLR, snr, nsnrIdx);

            //print_byte((UWORD8 *)pLLR, rate_match_request.G, "AWGN OUT: ");
            
            for(int i = 0; i < MAX_WIN_NUM; i++)
            {
                if(0 == (K % nWinSize[i]))
                {
                    /* 9. Rate dematch */
                    de_rm_request.pdmout = (UWORD8*)pLLR;
                    de_rm_request.k0withoutnull = 0;
                    de_rm_request.ncb = (K + 4) * 3;
                    de_rm_request.e = rate_match_request.G;
                    de_rm_request.isretx = 0;
                    de_rm_request.isinverted = 0;
                    de_rm_response.pharqbuffer = pTmpBuf1;
                    de_rm_response.pinteleavebuffer = pTmpBuf2;
                    de_rm_response.pharqout = pDeRmOut;
                    bblib_rate_match_ul(&de_rm_request, &de_rm_response, i);

                    //print_byte(pDeRmOut, rate_match_request.G, " DERM OUT: ");
                    
                    /* 10. decoder */
                    dec_request.c = 1;
                    dec_request.k = K;
                    dec_request.k_idx = K/8-4;
                    dec_request.max_iter_num = 4;
                    dec_request.early_term_disable = 0;
                    dec_request.input = (int8_t*)de_rm_response.pharqout;
                    dec_response.output = pDecOut;
                    dec_response.ag_buf = (int8_t*)pTmpBuf3;
                    dec_response.cb_buf = (uint16_t *)pTmpBuf4;
                    bblib_turbo_decoder(&dec_request, &dec_response, i);

                    ErrPktsIdx = PERestimation(K-CRC_LEN, pDecOut, pEnDataInBak);                    
                    nErrPkts[i] += ErrPktsIdx;

                    /*if((ErrPktsIdx != 0) || (dec_response.crc_status != 1))
                    {
                        printf("snrIdx %d pkgidx %d win %d err %d crc %d\n", nsnrIdx, nPktIdx, (1<<i)*8, ErrPktsIdx, dec_response.crc_status);
                        print_byte(pEnDataInBak, (K-CRC_LEN)/8, "REF: ");
                        print_byte(pDecOut, (K-CRC_LEN)/8, "OUT: ");
                        //print_bit(pEnDataInBak, (K-CRC_LEN)/8, "REF: ");
                        //print_bit(pDecOut, (K-CRC_LEN)/8, "OUT: ");
                    }*/
                    /*if((ErrPktsIdx != 0) && (dec_response.crc_status == 1))
                    {
                        printf("snrIdx %d pkgidx %d win %d err %d crc %d\n", nsnrIdx, nPktIdx, (1<<i)*8, ErrPktsIdx, dec_response.crc_status);
                        print_byte(pEnDataInBak, (K-CRC_LEN)/8, "REF: ");
                        print_byte(pDecOut, (K-CRC_LEN)/8, "OUT: ");
                    }*/
                }
                else
                {
                    nErrPkts[i] += -1;
                }
            } 
        }

        for(int i = 0; i < MAX_WIN_NUM; i++)
        {
            PER[nsnrIdx][i] = (double)nErrPkts[i]/(double)MAX_PKT_NUM;
        }
    }

    memcpy(pBler, PER, sizeof(PER));
    
    printf("calc SNR: ");

    for(nsnrIdx = 0; nsnrIdx < SNR_CNT; nsnrIdx++)
    {
        printf("%.2f ", gSnrCalc[nsnrIdx]/(float)MAX_PKT_NUM);
    }
    printf("\n");
    

    printf("\n\n\n[TBS = %d, SNR: %.1f ~ %.1f, %d Packets]\n\n", K-CRC_LEN, snrStart, snrStop, MAX_PKT_NUM);

    printf("   SNR ");

    for(nsnrIdx = snrStart; nsnrIdx < snrStop + 1; nsnrIdx++)
    {
        printf("%3d ", nsnrIdx);
    }
    printf("\n");
    
    for(int i = 0; i < MAX_WIN_NUM; i++)
    {
        if(PER[0][i] >= 0)
        {
            printf("WIN_%02d ", (1<<i)*8);

            for(nsnrIdx = 0; nsnrIdx < SNR_CNT; nsnrIdx++)
            {
                printf("%3d ", (int)(PER[nsnrIdx][i]*100.0));
            }

            printf("\n");
        }        
    }
    
    releaseTurbo();
    
    return 0;
}
