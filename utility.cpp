#include"utility.h"
#include<stdio.h>
#include<string.h>
#include <sys/stat.h>


void storeData(char *pData, const char *pFile, int Len)
{
    FILE* outFile;
    outFile = fopen(pFile, "ab+");
    fwrite(pData, Len, 1, outFile);
    fflush(outFile);
    fclose(outFile);
}

void loadData(char *pData, const char *pFile, int Len)
{
    FILE* outFile;
    outFile = fopen(pFile, "rb");
    //fseek(outFile,offset,0);
    fread(pData, Len, 1, outFile);
    fclose(outFile);
}

UWORD32 get_ms_cycles(void)
{
    FILE *fd = 0;
    UWORD32 cycles = 0;
    FLOAT32 freqf = 0;
    WORD8 line[1024];

    fd = fopen("/proc/cpuinfo", "r");
    if(fd == NULL)
    {
        fprintf(stderr, "get_ms_cycles: cannot open file: /proc/cpuinfo\n");
        exit(-1);
    }
    while(fgets(line, 1024, fd) != NULL)
    {
        if(sscanf(line, "cpu MHz\t: %f", &freqf) == 1)
        {
            cycles = (UWORD32)(freqf * 1000UL);
            break;
        }
    }
    fclose(fd);
    printf("The running CPU 1ms cycles = %d\n",cycles);
    return cycles;
}

//compare memory from a to b with number of num_bits bits. Return 0 if identical
int memcmp_bits(const void* a, const void* b, int num_bits)
{
    int res = 0;
    int num_words = num_bits/32;
    int left_bits = num_bits - num_words*32;
    int i;
    unsigned int *aa = (unsigned int*)a;
    unsigned int *bb = (unsigned int*)b;

    for(i=0; i<num_words; i++){
        if (aa[i] != bb[i]){
            res = -1;
#if 1
            printf("    MISMATCH @ {%d} : V [%08x] : M [%08x]\n", i, bb[i], aa[i]);
#endif
            goto ret;
        }
    }

    if( left_bits !=0 ){
        unsigned int a_bits = aa[num_words];
        unsigned int b_bits = bb[num_words];
        for(i=0; i<left_bits; i++){
            unsigned int a_bit = (a_bits >> i ) & 0x1;
            unsigned int b_bit = (b_bits >> i ) & 0x1;
            if( a_bit != b_bit ){
                res = -1;
                goto ret;
            }
        }
    }
ret:
    return res;
}

#define DUMP_DATA

#ifdef DUMP_DATA
char fileStoreBuffer[100];

#define PATH_DEBUG_STORE "./data/%d/"

void mkstore_dir(char *pBuf, char *pFile, int32_t ite)
{
    memset(pBuf, 0, 100);
    // bit remap output
    snprintf(pBuf, 100, PATH_DEBUG_STORE, ite);


    struct stat st = {0};
    if (stat(pBuf, &st) == -1) {
        mkdir(pBuf, 0777);
    }

    strcat(pBuf, pFile);
    printf("store data %s\n", pBuf);
}


void byte_print(int32_t ite, int16_t *pData, char *pFile, int32_t nLen) {

    mkstore_dir(fileStoreBuffer, pFile, ite);
    
    FILE* outFile;
    outFile = fopen(fileStoreBuffer, "w+");
    if (outFile)
    {
        for (uint32_t i = 0 ;i< nLen ;i++)
        {
            fprintf(outFile,"%d \n",pData[i]);
        }
        fclose(outFile);
    }
    else
    {
        printf("could not open %s file\n", fileStoreBuffer);
    }
}
void byte_print(int32_t ite, int8_t *pData, char *pFile, int32_t nLen) {

    mkstore_dir(fileStoreBuffer, pFile, ite);
    FILE* outFile;
    outFile = fopen(fileStoreBuffer, "w+");
    if (outFile)
    {
        for (uint32_t i = 0 ;i< nLen ;i++)
        {
            fprintf(outFile,"%d \n",pData[i]);
        }
        fclose(outFile);
    }
    else
    {
        printf("could not open %s file\n", fileStoreBuffer);
    }
}

void byte_print_multi_win(int32_t ite, int8_t *pData, char *pFile, int32_t nLen) {

    mkstore_dir(fileStoreBuffer, pFile, ite);
    FILE* outFile;
    outFile = fopen(fileStoreBuffer, "w+");
    if (outFile)
    {
        auto pDataOffset = pData;
        for (uint32_t i = 0; i < nLen ; i += 16)
        {
            for (uint32_t j = 0; j < 16 ; j += 1)
                fprintf(outFile,"%4d",pDataOffset[j]);
            fprintf(outFile,"\n");
            pDataOffset += 48;
        }
        fclose(outFile);
    }
    else
    {
        printf("could not open %s file\n", fileStoreBuffer);
    }
}

void ymm_print(__m128i data) {
    int8_t *pData = reinterpret_cast<int8_t *> (&data);
    for (uint16_t i = 0; i < 16; i++) {
        printf("%4d", pData[i]);
    }
    printf("\n");
}

#else
void byte_print(int32_t ite, int16_t *pData, char *pFile, int32_t nLen) {}

void byte_print(int32_t ite, int8_t *pData, char *pFile, int32_t nLen) {}

void byte_print_multi_win(int32_t ite, int8_t *pData, char *pFile, int32_t nLen) {}

void ymm_print(__m128i data) {}
#endif
