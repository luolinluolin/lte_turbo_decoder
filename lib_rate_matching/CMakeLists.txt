#######################################################################
#
# <COPYRIGHT_TAG>
#
#######################################################################

# Kernel specific flags
# Call add_compile_options' for EACH option
if(WIN32)
  # windows options
  add_compile_options("")
else()
  # linux options
  add_compile_options("")
  add_compile_options("-fPIC")
endif()

# Kernel sources
set (KernelSrcs
  phy_rate_match.cpp
  phy_rate_match_avx2.cpp
  phy_rate_match_sse.cpp
  phy_rate_match_sse_short.cpp
  phy_rate_match_sse_k6144.cpp
  phy_de_rate_match_avx2.cpp
  phy_de_rate_match_avx512.cpp
)

# Kernel includes (public only)
set (KernelIncs
  phy_rate_match.h
)


# NOTE: Enclosing variables in quotes forms a single argument parameter to allow
# for more than one file to be present.
ADD_KERNEL("${KernelSrcs}" "${KernelIncs}")
