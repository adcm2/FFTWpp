include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG main
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(Tests
               OdrInclude.cpp
               TestMain.cpp
               Tests.cpp)
target_link_libraries(Tests PRIVATE FFTWpp gtest)
include(GoogleTest)
gtest_discover_tests(Tests)
