#!/bin/make

CXX = icpx
# CXX = g++

IPPLIB = /opt/intel/oneapi/ipp/latest/lib/intel64

CFLAGS = -g -fPIC -O3 
CFLAGS += -xcore-avx512
CFLAGS += -D_BBLIB_AVX512_ -D_BBLIB_SSE4_2_
CFLAGS += -Wwritable-strings

LFLAGS = -lstdc++ -pthread -lrt



SRC = utility.cpp \
      sdk_version.cpp \
      divide.cpp \
      bblib_common.cpp \
      lib_crc/phy_crc.cpp \
      lib_crc/phy_crc_sse.cpp \
      lib_crc/phy_crc_avx512.cpp \
      lib_crc/phy_crc_snc.cpp \
      lib_rate_matching/phy_rate_match.cpp \
      lib_rate_matching/phy_rate_match_avx2.cpp \
      lib_rate_matching/phy_rate_match_sse.cpp \
      lib_rate_matching/phy_rate_match_sse_short.cpp \
      lib_rate_matching/phy_rate_match_sse_k6144.cpp \
      lib_rate_matching/phy_de_rate_match_avx2.cpp \
      lib_rate_matching/phy_de_rate_match_avx512.cpp \
      lib_turbo/phy_turbo.cpp \
      lib_turbo/phy_turbo_encoder_avx2.cpp \
      lib_turbo/phy_turbo_encoder_avx512.cpp \
      lib_turbo/phy_turbo_decoder_16windows_sse.cpp \
      lib_turbo/phy_turbo_decoder_32windows_avx2.cpp \
      lib_turbo/phy_turbo_decoder_64windows_avx512.cpp \
      lib_turbo/phy_turbo_encoder_sse.cpp \
      lib_turbo/phy_turbo_decoder_8windows_sse.cpp \
      lib_turbo/phy_turbo_fast_interleave_sse.cpp \
      lib_turbo/phy_turbo_decoder_MakeTable.cpp \
      turbo_dec_avx.cpp

INCLUDES  = -I ./lib_rate_matching \
           -I ./lib_scramble \
           -I ./lib_turbo \
           -I ./lib_crc \
	     -I ./

INCLUDES  += -I $(MATLAB)/extern/include \
            -I $(MATLAB)/simulink/include

LIBS = -shared -Wl,--version-script,$(MATLAB)/extern/lib/glnxa64/mexFunction.map -Wl,--no-undefined
LIBS += -Wl,-rpath-link,$(MATLAB)/bin/glnxa64 -L$(MATLAB)/bin/glnxa64 -lmx -lmex -lmat -lipps
LIBS += -lpthread -lm -ldl

OBJS = $(SRC:.cpp=.o)


TAR = turbo_dec_avx.mexa64 

all: $(TAR)

$(TAR): $(OBJS)
	$(CXX) $(LFLAGS) $(INCLUDES) $(OBJS) $(LIBS) -o $@
.cpp.o:
	$(CXX) $(INCLUDES) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJS) $(tar) *.mexa64
