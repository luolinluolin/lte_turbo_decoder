#ifndef _UTILITY_H
#define _UTILITY_H
#include"common_typedef.h"
#include<stdlib.h>
#include <malloc.h>
#include <unistd.h>
#include <stdint.h>
#include "immintrin.h"

#define OUT_FLAG 0
void storeData(char *pData, const char *pFile, int Len);
void loadData(char *pData, const char *pFile, int Len);
UWORD32 get_ms_cycles(void);
int memcmp_bits(const void* a, const void* b, int num_bits);

void byte_print(int32_t ite, int16_t *pData, char *pFile, int32_t nLen);
void byte_print(int32_t ite, int8_t *pData, char *pFile, int32_t nLen);
void byte_print_multi_win(int32_t ite, int8_t *pData, char *pFile, int32_t nLen);
void ymm_print(__m128i data);
#endif