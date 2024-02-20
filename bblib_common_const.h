/*******************************************************************************
*
* <COPYRIGHT_TAG>
*
*******************************************************************************/

/**
 * @file bblib_common_const.h
 * @brief This header file defines common global constants uses throughout the
 * bblib libraries.
 */

#ifndef _BBLIB_COMMON_CONST_
#define _BBLIB_COMMON_CONST_

#ifndef likely
#define likely(x) __builtin_expect(!!(x), 1)
#endif

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

#ifndef RUP512B
#define RUP512B(x) (((x)+511)&(~511))
#endif
#ifndef RUP256B
#define RUP256B(x) (((x)+255)&(~255))
#endif
#ifndef RUP128B
#define RUP128B(x) (((x)+127)&(~127))
#endif
#ifndef RUP64B
#define RUP64B(x) (((x)+63)&(~63))
#endif
#ifndef RUP32B
#define RUP32B(x) (((x)+31)&(~31))
#endif
#ifndef RUP16B
#define RUP16B(x) (((x)+15)&(~15))
#endif
#ifndef RUP8B
#define RUP8B(x)  (((x)+7)&(~7))
#endif
#ifndef RUP4B
#define RUP4B(x)  (((x)+3)&(~3))
#endif
#ifndef RUP2B
#define RUP2B(x)  (((x)+1)&(~1))
#endif

#ifndef PI
#define PI ((float) 3.14159265358979323846)
#endif

#ifndef PI_double
#define PI_double ((double) 3.14159265358979323846)
#endif

#ifndef MAX_MU_NUM
#define MAX_MU_NUM (5)
#endif

#ifndef N_FFT_SIZE_MU0_5MHZ
#define N_FFT_SIZE_MU0_5MHZ (512)
#endif

#ifndef N_FFT_SIZE_MU0_10MHZ
#define N_FFT_SIZE_MU0_10MHZ (1024)
#endif

#ifndef N_FFT_SIZE_MU0_15MHZ
#define N_FFT_SIZE_MU0_15MHZ (1536)
#endif

#ifndef N_FFT_SIZE_MU0_20MHZ_AND_25MHZ
#define N_FFT_SIZE_MU0_20MHZ_AND_25MHZ (2048)
#endif

#ifndef N_FFT_SIZE_MU0_30MHZ_AND_35MHZ
#define N_FFT_SIZE_MU0_30MHZ_AND_35MHZ (3072)
#endif

#ifndef N_FFT_SIZE_MU0_40MHZ_AND_50MHZ
#define N_FFT_SIZE_MU0_40MHZ_AND_50MHZ (4096)
#endif

#ifndef N_FFT_SIZE_MU1_5MHZ
#define N_FFT_SIZE_MU1_5MHZ (256)
#endif

#ifndef N_FFT_SIZE_MU1_10MHZ
#define N_FFT_SIZE_MU1_10MHZ (512)
#endif

#ifndef N_FFT_SIZE_MU1_15MHZ
#define N_FFT_SIZE_MU1_15MHZ (768)
#endif

#ifndef N_FFT_SIZE_MU1_20MHZ_AND_25MHZ
#define N_FFT_SIZE_MU1_20MHZ_AND_25MHZ (1024)
#endif

#ifndef N_FFT_SIZE_MU1_30MHZ_AND_35MHZ
#define N_FFT_SIZE_MU1_30MHZ_AND_35MHZ (1536)
#endif

#ifndef N_FFT_SIZE_MU1_40MHZ_45MHZ_AND_50MHZ
#define N_FFT_SIZE_MU1_40MHZ_45MHZ_AND_50MHZ (2048)
#endif

#ifndef N_FFT_SIZE_MU1_60MHZ_AND_70MHZ
#define N_FFT_SIZE_MU1_60MHZ_AND_70MHZ (3072)
#endif

#ifndef N_FFT_SIZE_MU1_80MHZ_90MHZ_AND_100MHZ
#define N_FFT_SIZE_MU1_80MHZ_90MHZ_AND_100MHZ (4096)
#endif

#ifndef N_FFT_SIZE_MU2_10MHZ
#define N_FFT_SIZE_MU2_10MHZ (256)
#endif

#ifndef N_FFT_SIZE_MU2_15MHZ
#define N_FFT_SIZE_MU2_15MHZ (384)
#endif

#ifndef N_FFT_SIZE_MU2_20MHZ_AND_25MHZ
#define N_FFT_SIZE_MU2_20MHZ_AND_25MHZ (512)
#endif

#ifndef N_FFT_SIZE_MU2_30MHZ_AND_35MHZ
#define N_FFT_SIZE_MU2_30MHZ_AND_35MHZ (768)
#endif

#ifndef N_FFT_SIZE_MU2_40MHZ_45MHZ_AND_50MHZ
#define N_FFT_SIZE_MU2_40MHZ_45MHZ_AND_50MHZ (1024)
#endif

#ifndef N_FFT_SIZE_MU2_60MHZ_AND_70MHZ
#define N_FFT_SIZE_MU2_60MHZ_AND_70MHZ (1536)
#endif

#ifndef N_FFT_SIZE_MU2_80MHZ_90MHZ_AND_100MHZ
#define N_FFT_SIZE_MU2_80MHZ_90MHZ_AND_100MHZ (2048)
#endif

#ifndef N_FFT_SIZE_MU3_50MHZ
#define N_FFT_SIZE_MU3_50MHZ (512)
#endif

#ifndef N_FFT_SIZE_MU3_100MHZ
#define N_FFT_SIZE_MU3_100MHZ (1024)
#endif

#ifndef N_FFT_SIZE_MU3_200MHZ
#define N_FFT_SIZE_MU3_200MHZ (2048)
#endif

#ifndef N_FFT_SIZE_MU3_400MHZ
#define N_FFT_SIZE_MU3_400MHZ (4096)
#endif

#ifndef N_MAX_CP_MU0_5MHZ
#define N_MAX_CP_MU0_5MHZ (40)
#endif

#ifndef N_MAX_CP_MU0_10MHZ
#define N_MAX_CP_MU0_10MHZ (80)
#endif

#ifndef N_MAX_CP_MU0_15MHZ
#define N_MAX_CP_MU0_15MHZ (120)
#endif

#ifndef N_MAX_CP_MU0_20MHZ_AND_25MHZ
#define N_MAX_CP_MU0_20MHZ_AND_25MHZ (160)
#endif

#ifndef N_MAX_CP_MU0_30MHZ_AND_35MHZ
#define N_MAX_CP_MU0_30MHZ_AND_35MHZ (240)
#endif

#ifndef N_MAX_CP_MU0_40MHZ_AND_50MHZ
#define N_MAX_CP_MU0_40MHZ_AND_50MHZ (320)
#endif

#ifndef N_MAX_CP_MU1_5MHZ
#define N_MAX_CP_MU1_5MHZ (22)
#endif

#ifndef N_MAX_CP_MU1_10MHZ
#define N_MAX_CP_MU1_10MHZ (44)
#endif

#ifndef N_MAX_CP_MU1_15MHZ
#define N_MAX_CP_MU1_15MHZ (66)
#endif

#ifndef N_MAX_CP_MU1_20MHZ_AND_25MHZ
#define N_MAX_CP_MU1_20MHZ_AND_25MHZ (88)
#endif

#ifndef N_MAX_CP_MU1_30MHZ_AND_35MHZ
#define N_MAX_CP_MU1_30MHZ_AND_35MHZ (132)
#endif

#ifndef N_MAX_CP_MU1_40MHZ_45MHZ_AND_50MHZ
#define N_MAX_CP_MU1_40MHZ_45MHZ_AND_50MHZ (176)
#endif

#ifndef N_MAX_CP_MU1_60MHZ_AND_70MHZ
#define N_MAX_CP_MU1_60MHZ_AND_70MHZ (264)
#endif

#ifndef N_MAX_CP_MU1_80MHZ_90MHZ_AND_100MHZ
#define N_MAX_CP_MU1_80MHZ_90MHZ_AND_100MHZ (352)
#endif

#ifndef N_MAX_CP_MU2_10MHZ
#define N_MAX_CP_MU2_10MHZ (26)
#endif

#ifndef N_MAX_CP_MU2_15MHZ
#define N_MAX_CP_MU2_15MHZ (39)
#endif

#ifndef N_MAX_CP_MU2_20MHZ_AND_25MHZ
#define N_MAX_CP_MU2_20MHZ_AND_25MHZ (52)
#endif

#ifndef N_MAX_CP_MU2_30MHZ_AND_35MHZ
#define N_MAX_CP_MU2_30MHZ_AND_35MHZ (78)
#endif

#ifndef N_MAX_CP_MU2_40MHZ_45MHZ_AND_50MHZ
#define N_MAX_CP_MU2_40MHZ_45MHZ_AND_50MHZ (104)
#endif

#ifndef N_MAX_CP_MU2_60MHZ_AND_70MHZ
#define N_MAX_CP_MU2_60MHZ_AND_70MHZ (156)
#endif

#ifndef N_MAX_CP_MU2_80MHZ_90MHZ_AND_100MHZ
#define N_MAX_CP_MU2_80MHZ_90MHZ_AND_100MHZ (208)
#endif

#ifndef N_MAX_CP_MU3_50MHZ
#define N_MAX_CP_MU3_50MHZ (68)
#endif

#ifndef N_MAX_CP_MU3_100MHZ
#define N_MAX_CP_MU3_100MHZ (136)
#endif

#ifndef N_MAX_CP_MU3_200MHZ
#define N_MAX_CP_MU3_200MHZ (272)
#endif

#ifndef N_MAX_CP_MU3_400MHZ
#define N_MAX_CP_MU3_400MHZ (544)
#endif

#ifndef N_MIN_CP_MU0_5MHZ
#define N_MIN_CP_MU0_5MHZ (36)
#endif

#ifndef N_MIN_CP_MU0_10MHZ
#define N_MIN_CP_MU0_10MHZ (72)
#endif

#ifndef N_MIN_CP_MU0_15MHZ
#define N_MIN_CP_MU0_15MHZ (108)
#endif

#ifndef N_MIN_CP_MU0_20MHZ_AND_25MHZ
#define N_MIN_CP_MU0_20MHZ_AND_25MHZ (144)
#endif

#ifndef N_MIN_CP_MU0_30MHZ_AND_35MHZ
#define N_MIN_CP_MU0_30MHZ_AND_35MHZ (216)
#endif

#ifndef N_MIN_CP_MU0_40MHZ_AND_50MHZ
#define N_MIN_CP_MU0_40MHZ_AND_50MHZ (288)
#endif

#ifndef N_MIN_CP_MU1_5MHZ
#define N_MIN_CP_MU1_5MHZ (18)
#endif

#ifndef N_MIN_CP_MU1_10MHZ
#define N_MIN_CP_MU1_10MHZ (36)
#endif

#ifndef N_MIN_CP_MU1_15MHZ
#define N_MIN_CP_MU1_15MHZ (54)
#endif

#ifndef N_MIN_CP_MU1_20MHZ_AND_25MHZ
#define N_MIN_CP_MU1_20MHZ_AND_25MHZ (72)
#endif

#ifndef N_MIN_CP_MU1_30MHZ_AND_35MHZ
#define N_MIN_CP_MU1_30MHZ_AND_35MHZ (108)
#endif

#ifndef N_MIN_CP_MU1_40MHZ_45MHZ_AND_50MHZ
#define N_MIN_CP_MU1_40MHZ_45MHZ_AND_50MHZ (144)
#endif

#ifndef N_MIN_CP_MU1_60MHZ_AND_70MHZ
#define N_MIN_CP_MU1_60MHZ_AND_70MHZ (216)
#endif

#ifndef N_MIN_CP_MU1_80MHZ_90MHZ_AND_100MHZ
#define N_MIN_CP_MU1_80MHZ_90MHZ_AND_100MHZ (288)
#endif

#ifndef N_MIN_CP_MU2_10MHZ
#define N_MIN_CP_MU2_10MHZ (18)
#endif

#ifndef N_MIN_CP_MU2_15MHZ
#define N_MIN_CP_MU2_15MHZ (27)
#endif

#ifndef N_MIN_CP_MU2_20MHZ_AND_25MHZ
#define N_MIN_CP_MU2_20MHZ_AND_25MHZ (36)
#endif

#ifndef N_MIN_CP_MU2_30MHZ_AND_35MHZ
#define N_MIN_CP_MU2_30MHZ_AND_35MHZ (54)
#endif

#ifndef N_MIN_CP_MU2_40MHZ_45MHZ_AND_50MHZ
#define N_MIN_CP_MU2_40MHZ_45MHZ_AND_50MHZ (72)
#endif

#ifndef N_MIN_CP_MU2_60MHZ_AND_70MHZ
#define N_MIN_CP_MU2_60MHZ_AND_70MHZ (108)
#endif

#ifndef N_MIN_CP_MU2_80MHZ_90MHZ_AND_100MHZ
#define N_MIN_CP_MU2_80MHZ_90MHZ_AND_100MHZ (144)
#endif

#ifndef N_MIN_CP_MU3_50MHZ
#define N_MIN_CP_MU3_50MHZ (36)
#endif

#ifndef N_MIN_CP_MU3_100MHZ
#define N_MIN_CP_MU3_100MHZ (72)
#endif

#ifndef N_MIN_CP_MU3_200MHZ
#define N_MIN_CP_MU3_200MHZ (144)
#endif

#ifndef N_MIN_CP_MU3_400MHZ
#define N_MIN_CP_MU3_400MHZ (288)
#endif

#ifndef N_FULLBAND_SC_MU0_5MHZ
#define N_FULLBAND_SC_MU0_5MHZ (300)
#endif

#ifndef N_FULLBAND_SC_MU0_10MHZ
#define N_FULLBAND_SC_MU0_10MHZ (624)
#endif

#ifndef N_FULLBAND_SC_MU0_20MHZ
#define N_FULLBAND_SC_MU0_20MHZ (1272)
#endif

#ifndef N_FULLBAND_SC_MU0_40MHZ
#define N_FULLBAND_SC_MU0_40MHZ (2592)
#endif

#ifndef N_FULLBAND_SC_MU1_10MHZ
#define N_FULLBAND_SC_MU1_10MHZ (288)
#endif

#ifndef N_FULLBAND_SC_MU1_20MHZ
#define N_FULLBAND_SC_MU1_20MHZ (612)
#endif

#ifndef N_FULLBAND_SC_MU1_40MHZ
#define N_FULLBAND_SC_MU1_40MHZ (1272)
#endif

#ifndef N_FULLBAND_SC_MU1_50MHZ
#define N_FULLBAND_SC_MU1_50MHZ (1596)
#endif

#ifndef N_FULLBAND_SC_MU1_60MHZ
#define N_FULLBAND_SC_MU1_60MHZ (1944)
#endif

#ifndef N_FULLBAND_SC_MU1_70MHZ
#define N_FULLBAND_SC_MU1_70MHZ (2268)
#endif

#ifndef N_FULLBAND_SC_MU1_100MHZ
#define N_FULLBAND_SC_MU1_100MHZ (3276)
#endif

#ifndef N_FULLBAND_SC_MU3
#define N_FULLBAND_SC_MU3 (792)
#endif

#ifndef N_SAMPLE_RATE_MU0_40MHZ_AND_50MHZ
#define N_SAMPLE_RATE_MU0_40MHZ_AND_50MHZ (61.44*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU0_30MHZ_AND_35MHZ
#define N_SAMPLE_RATE_MU0_30MHZ_AND_35MHZ (46.08*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU0_20MHZ_AND_25MHZ
#define N_SAMPLE_RATE_MU0_20MHZ_AND_25MHZ (30.72*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU0_15MHZ
#define N_SAMPLE_RATE_MU0_15MHZ (23.04*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU0_10MHZ
#define N_SAMPLE_RATE_MU0_10MHZ (15.36*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU0_5MHZ
#define N_SAMPLE_RATE_MU0_5MHZ (7.68*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU1_5MHZ
#define N_SAMPLE_RATE_MU1_5MHZ (7.68*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU1_10MHZ
#define N_SAMPLE_RATE_MU1_10MHZ (15.36*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU1_15MHZ
#define N_SAMPLE_RATE_MU1_15MHZ (23.04*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU1_20MHZ_AND_25MHZ
#define N_SAMPLE_RATE_MU1_20MHZ_AND_25MHZ (30.72*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU1_30MHZ_AND_35MHZ
#define N_SAMPLE_RATE_MU1_30MHZ_AND_35MHZ (46.08*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU1_40MHZ_45MHZ_AND_50MHZ
#define N_SAMPLE_RATE_MU1_40MHZ_45MHZ_AND_50MHZ (61.44*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU1_60MHZ_AND_70MHZ
#define N_SAMPLE_RATE_MU1_60MHZ_AND_70MHZ (92.16*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU1_80MHZ_90MHZ_AND_100MHZ
#define N_SAMPLE_RATE_MU1_80MHZ_90MHZ_AND_100MHZ (122.88*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU2_10MHZ
#define N_SAMPLE_RATE_MU2_10MHZ (15.36*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU2_15MHZ
#define N_SAMPLE_RATE_MU2_15MHZ (23.04*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU2_20MHZ_AND_25MHZ
#define N_SAMPLE_RATE_MU2_20MHZ_AND_25MHZ (30.72*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU2_30MHZ_AND_35MHZ
#define N_SAMPLE_RATE_MU2_30MHZ_AND_35MHZ (46.08*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU2_40MHZ_45MHZ_AND_50MHZ
#define N_SAMPLE_RATE_MU2_40MHZ_45MHZ_AND_50MHZ (61.44*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU2_60MHZ_AND_70MHZ
#define N_SAMPLE_RATE_MU2_60MHZ_AND_70MHZ (92.16*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU2_80MHZ_90MHZ_AND_100MHZ
#define N_SAMPLE_RATE_MU2_80MHZ_90MHZ_AND_100MHZ (122.88*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU3_50MHZ
#define N_SAMPLE_RATE_MU3_50MHZ (61.44*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU3_100MHZ
#define N_SAMPLE_RATE_MU3_100MHZ (122.88*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU3_200MHZ
#define N_SAMPLE_RATE_MU3_200MHZ (245.76*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU3_400MHZ
#define N_SAMPLE_RATE_MU3_400MHZ (491.52*1000*1000)
#endif

#ifndef N_DMRS_TYPE1_SC_PER_RB
#define N_DMRS_TYPE1_SC_PER_RB (6)
#endif

#ifndef N_DMRS_TYPE2_SC_PER_RB
#define N_DMRS_TYPE2_SC_PER_RB (4)
#endif

#ifndef N_DMRS_TYPE1_DELTA
#define N_DMRS_TYPE1_DELTA (2)
#endif

#ifndef N_DMRS_TYPE2_DELTA
#define N_DMRS_TYPE2_DELTA (3)
#endif

#ifndef MAX_NUM_OF_DELTA
#define MAX_NUM_OF_DELTA (3)
#endif

#ifndef DMRS_TYPE1_MAX_PORT_NUM
#define DMRS_TYPE1_MAX_PORT_NUM (8)
#endif

#ifndef DMRS_TYPE2_MAX_PORT_NUM
#define DMRS_TYPE2_MAX_PORT_NUM (12)
#endif

#ifndef DMRS_TYPE1_SINGLE_DMRS_MAX_PORT_NUM
#define DMRS_TYPE1_SINGLE_DMRS_MAX_PORT_NUM (4)
#endif

#ifndef DMRS_TYPE2_SINGLE_DMRS_MAX_PORT_NUM
#define DMRS_TYPE2_SINGLE_DMRS_MAX_PORT_NUM (6)
#endif

#ifndef DMRS_MAX_PORT_NUM_PER_CDM
#define DMRS_MAX_PORT_NUM_PER_CDM (4)
#endif

#ifndef RNN_FP16_SCALE
#define RNN_FP16_SCALE (4)
#endif

#endif /* _BBLIB_COMMON_CONST_ */



