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
#include <stdlib.h>
#include <malloc.h>
#include <immintrin.h> /* AVX */
#include <assert.h>
#include "phy_rate_match.h"
#include "phy_turbo.h"
#include "matrix.h"
#include "mex.h"


#define SOFT_VALUE 16
#define SCALE_NOISE 2
#define MAX_TB_SIZE 6120
#define CRC_LEN 24
#define MAX_K MAX_TB_SIZE + CRC_LEN
#define MAX_WIN_NUM 4
#define ALIGN_SIZE  512
#define _aligned_malloc(x,y) memalign(y,x)
//#define __align(x) __attribute__((aligned(x)))


typedef unsigned char UWORD8;
typedef unsigned short UWORD16;


static __align(ALIGN_SIZE) UWORD8 gDecOut[MAX_K/8] = {0}; 
static UWORD8 *pDecOut = NULL;
static UWORD8 *pTmpBuf3 = NULL;
static UWORD8 *pTmpBuf4 = NULL;
static UWORD8 *pDeRmOut = NULL;
static UWORD8 *pData = NULL;

static void initialTurbo()
{
    pData = (UWORD8 *)_aligned_malloc(3*(MAX_K + 4), ALIGN_SIZE);
    assert(pData != NULL);
    pTmpBuf3 = (UWORD8 *)_aligned_malloc(6528 * 16,ALIGN_SIZE);
    assert(pTmpBuf3 != NULL);
    pTmpBuf4 = (UWORD8 *)_aligned_malloc(MAX_K,ALIGN_SIZE);
    assert(pTmpBuf4 != NULL);
    pDeRmOut = (UWORD8 *)_aligned_malloc(3*(MAX_K + 4),ALIGN_SIZE);
    assert(pDeRmOut != NULL);
}

static void releaseTurbo()
{
    free(pData);
    pData = NULL;
    free(pTmpBuf3);
    pTmpBuf3 = NULL;
    free(pTmpBuf4);
    pTmpBuf4 = NULL;
    free(pDeRmOut);
    pDeRmOut = NULL;
}

static void byte_2_bit(const UWORD8 *pIn, int len, double *pOut)
{
    int n=0;

    for(int i=0; i<len; i++)
    {
        for(int j=0; j<8; j++)
        {
            pOut[n++] = (double)((pIn[i] >> j) & 1);
        }
    }
}

static void set_neg1(double *pOut, int len)
{
    for(int i=0; i<len; i++)
    {
        pOut[i] = -1.0;
    }
}

static int get_k_idx(int K)
{
    if(K <= 512)
    {
        return K / 8 - 4;
    }
    else if(K <= 1024)
    {
        return (K - 528) / 16 + 61;
    }
    else if(K <= 2048)
    {
        return (K - 1056) / 32 + 93;
    }
    else
    {
        return (K - 2112) / 64 + 125;
    }
}

static int turbo_decoder_core(int K, int nIterNum, UWORD8 *pDataIn, double *pDataOut, double *pCrc, int nWinIdx)
{
    struct bblib_turbo_adapter_ul_request adapter_request;
    struct bblib_turbo_adapter_ul_response adapter_response;
    struct bblib_turbo_decoder_request dec_request;
    struct bblib_turbo_decoder_response dec_response;

    /* 9. Rate dematch */
    adapter_request.ncb = (K + 4) * 3;
    adapter_request.isinverted = 0;
    adapter_request.pinteleavebuffer = pDataIn;
    adapter_response.pharqout = pDeRmOut;
    bblib_turbo_adapter_ul_avx512(&adapter_request, &adapter_response, nWinIdx);
    
    /* 10. decoder */
    dec_request.c = 1;
    dec_request.k = K;
    dec_request.k_idx = get_k_idx(K);
    dec_request.max_iter_num = nIterNum;
    dec_request.early_term_disable = 0;
    dec_request.input = (int8_t*)adapter_response.pharqout;
    dec_response.output = gDecOut;
    dec_response.ag_buf = (int8_t*)pTmpBuf3;
    dec_response.cb_buf = (UWORD16 *)pTmpBuf4;
    bblib_turbo_decoder(&dec_request, &dec_response, nWinIdx);

    byte_2_bit(gDecOut, (K-CRC_LEN)/8,  pDataOut);

    pCrc[0] = dec_response.crc_status;

    return 0;
}


/* main function that interfaces with MATLAB */
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int K, IterNum, DataCnt;
    double *pK, *pTempDataIn, *pCrc, *pBitOut;
    

    if (nrhs != 4)
    {
        mexPrintf("error: nrhs(%d) wrong, must be 3!\n", nrhs);
        return;
    }
    
    K = (int)mxGetScalar(prhs[0]);
    IterNum = (int)mxGetScalar(prhs[1]);
    pTempDataIn = mxGetPr(prhs[2]);
    DataCnt = mxGetN(prhs[2]);
    int nWinIdx = (int)mxGetScalar(prhs[3]);
    
    plhs[0] = mxCreateDoubleMatrix(K-CRC_LEN, 1, mxREAL);
    pBitOut = mxGetPr(plhs[0]);
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
    pCrc = mxGetPr(plhs[1]);

    initialTurbo();

    for(int i = 0; i < DataCnt; i++)
    {
        pData[i] = (int)pTempDataIn[i];
    }
    
    turbo_decoder_core(K, IterNum, pData, pBitOut, pCrc, nWinIdx);

    releaseTurbo();
}

