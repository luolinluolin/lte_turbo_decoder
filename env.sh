export GTEST_ROOT=/opt/gtest/gtest-1.7.0
source /opt/intel/oneapi/setvars.sh --force
export PATH=$PATH:/opt/intel/oneapi/compiler/latest/bin/compiler/
export PATH=/opt/cmake-3.20.2/bin/:$PATH
export PATH=/opt/sde-internal-conf-9.3.0-2022-01-12-lin:$PATH
#source /opt/intel/system_studio_2020/bin/iccvars.sh intel64

export PATH=/opt/cmake-3.20.2/bin/:$PATH
export PATH=/usr/local/MATLAB/R2021a/bin:${PATH}
export CMAKE_BUILD_TYPE="Debug"

export MATLAB=/usr/local/MATLAB/R2021a/
export MATLAB_LIBRARY_DIR=${MATLAB}/bin/glnxa64

