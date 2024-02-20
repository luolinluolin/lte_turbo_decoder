/*******************************************************************************
*
* INTEL CONFIDENTIAL
* Copyright 2011-2012 Intel Corporation All Rights Reserved.
* 
* The source code contained or described herein and all documents related to the
* source code ("Material") are owned by Intel Corporation or its suppliers or
* licensors. Title to the Material remains with Intel Corporation or its
* suppliers and licensors. The Material may contain trade secrets and proprietary
* and confidential information of Intel Corporation and its suppliers and
* licensors, and is protected by worldwide copyright and trade secret laws and
* treaty provisions. No part of the Material may be used, copied, reproduced,
* modified, published, uploaded, posted, transmitted, distributed, or disclosed
* in any way without Intels prior express written permission.
* 
* No license under any patent, copyright, trade secret or other intellectual
* property right is granted to or conferred upon you by disclosure or delivery
* of the Materials, either expressly, by implication, inducement, estoppel or
* otherwise. Any license under such intellectual property rights must be
* express and approved by Intel in writing.
* 
* Unless otherwise agreed by Intel in writing, you may not remove or alter this
* notice or any other notice embedded in Materials by Intel or Intels suppliers
* or licensors in any way.
* 
*  version: TURBO_DECODER_OPT_HSW.L.0.1.0-12
*  version: TURBO_DECODER_OPT_HSW.L.0.1.0-12
*
*******************************************************************************/
/*******************************************************************************
* @file		- Common_typedef.h
* @brief	- This header file defines those data type both used by eNB and UE.
* @author	- Leifeng Ruan(leifeng.ruan@intel.com), Xuebin Yang(xuebin.yang@intel.com) Intel Labs China
*******************************************************************************/

/**********************************************************************
* Attentation and assumption:
*     LSBs are used when one variable needs only 2, 4 etc bits when there 
*     is no explicit declarition.
**********************************************************************/

#ifndef _COMMON_TYPEDEF_H_
#define _COMMON_TYPEDEF_H_

/** \brief Usage: BOOLEAN bool; */
//typedef bool BOOLEAN;

/** \brief Usage: UWORD8 u8Tmp; */
typedef unsigned char UWORD8;

/** \brief Usage: WORD8 i8Tmp; */
typedef char WORD8;

/** \brief Usage: UWORD16 u16Tmp; */
typedef unsigned short UWORD16;

/** \brief Usage: WORD16 i16Tmp; */
typedef short WORD16;

/** \brief Usage: UWORD32 u32Tmp; */
typedef unsigned int UWORD32;

/** \brief Usage: WORD32 i32Tmp; */
typedef int WORD32;


/** \brief Usage: WORD64 i64Tmp; */
typedef long WORD64;

/** \brief Usage: UWORD64 u64Tmp; */
typedef unsigned long long UWORD64;

/** \brief Usage: FLOAT32  floatTmp; */
typedef float FLOAT32;

/** \brief Usage: DOUBLE64 doubleTmp; */
typedef double DOUBLE64;

/** \brief Usage: COMPLEX16 cp16Tmp; */
typedef struct {
    WORD16 re;
    WORD16 im;
} COMPLEX16;
/** \value it may take 0,1 or 1,-1*/
typedef	signed short  Bits; 

#endif /* #ifndef _COMMON_TYPEDEF_H_ */
