#######################################################################
#
# <COPYRIGHT_TAG>
#
#######################################################################

# Kernel specific flags
# Call add_compile_options' for EACH option
if(WIN32)
  # windows options
  add_compile_options("/w")
else()
  # linux options
  add_compile_options("-w")
  add_compile_options("-fPIC")
endif()

# Kernel sources
set (KernelSrcs
  phy_crc.cpp
  phy_crc_sse.cpp
  phy_crc_avx512.cpp
  phy_crc_snc.cpp
)

# Kernel includes (public only)
set (KernelIncs
  phy_crc.h
)

# NOTE: Enclosing variables in quotes forms a single argument parameter to allow
# for more than one file to be present.
ADD_KERNEL("${KernelSrcs}" "${KernelIncs}")

