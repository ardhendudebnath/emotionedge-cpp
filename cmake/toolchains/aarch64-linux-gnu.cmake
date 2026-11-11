# Cross-compiles for 64-bit ARM Linux (Jetson-class boards, Raspberry Pi 5, ARM servers) with
# GCC's aarch64-linux-gnu cross toolchain. What the build runs for the target (test discovery,
# ctest) runs under QEMU user emulation when it is found.
#
#   cmake -S . -B build-arm64 -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch64-linux-gnu.cmake
#
# EE_CROSS_ROOT is where the toolchain is: /usr by default, as Debian and Ubuntu's
# g++-aarch64-linux-gnu and qemu-user-static install it. A copy of those packages unpacked
# elsewhere (dpkg -x, no root) works too; give its usr/ and, for a versioned compiler only,
# EE_CROSS_GCC_SUFFIX (e.g. -13).
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(EE_CROSS_ROOT "/usr" CACHE PATH "Root of the aarch64 cross toolchain: bin/aarch64-linux-gnu-g++ and the aarch64-linux-gnu/ sysroot")
set(EE_CROSS_GCC_SUFFIX "" CACHE STRING "Version suffix of the cross compilers, e.g. -13")
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES EE_CROSS_ROOT EE_CROSS_GCC_SUFFIX)

set(CMAKE_C_COMPILER "${EE_CROSS_ROOT}/bin/aarch64-linux-gnu-gcc${EE_CROSS_GCC_SUFFIX}")
set(CMAKE_CXX_COMPILER "${EE_CROSS_ROOT}/bin/aarch64-linux-gnu-g++${EE_CROSS_GCC_SUFFIX}")

# Libraries, headers and packages come from the target's sysroot only: a host (x86-64)
# yaml-cpp or GTest must not satisfy find_package, so FetchContent builds them for the target.
set(EE_CROSS_SYSROOT "${EE_CROSS_ROOT}/aarch64-linux-gnu")
set(CMAKE_FIND_ROOT_PATH "${EE_CROSS_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

find_program(EE_QEMU_AARCH64 NAMES qemu-aarch64-static qemu-aarch64 HINTS "${EE_CROSS_ROOT}/bin")
if(EE_QEMU_AARCH64)
    set(CMAKE_CROSSCOMPILING_EMULATOR "${EE_QEMU_AARCH64};-L;${EE_CROSS_SYSROOT}")
endif()
