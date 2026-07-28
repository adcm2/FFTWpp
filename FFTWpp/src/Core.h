/**
 * @file Core.h
 * @brief Core header for the FFTWpp library, providing C++ wrappers for FFTW3
 * functions.
 *
 * This file defines type-safe and precision-aware C++ wrappers for the core
 * functionality of the FFTW3 library. It includes:
 * - A custom STL allocator for memory alignment required by FFTW.
 * - Template-based plan creation functions for various transform types (DFT,
 * R2C, C2R, R2R) across 1D, 2D, 3D, and N-dimensional data.
 * - Overloaded functions that automatically select the correct FFTW precision
 * (float, double, long double) based on the data type.
 * - Wrapper functions for executing and destroying plans.
 */
#ifndef FFTWPP_CORE_GUARD_H
#define FFTWPP_CORE_GUARD_H

#include <cassert>
#include <complex>
#include <concepts>
#include <memory>
#include <variant>
#include <vector>

#include "NumericConcepts/Numeric.hpp"
#include "fftw3.h"

/**
 * @brief Main namespace for the FFTWpp C++ wrapper library.
 */
namespace FFTWpp {

//------------------------------------------------------//
//              Define some useful concepts             //
//------------------------------------------------------//

/**
 * @brief Concept to check if a type is a valid FFTW plan handle.
 * @tparam T The type to check.
 */
template <typename T>
concept IsPlan = std::same_as<T, fftwf_plan> or std::same_as<T, fftw_plan> or
                 std::same_as<T, fftwl_plan>;

/**
 * @brief Concept to ensure that the precision of a plan matches the precision
 * of the real data type.
 * @tparam PlanType The FFTW plan type (e.g., fftwf_plan).
 * @tparam Real The real data type (e.g., float, double).
 */
template <typename PlanType, typename Real>
concept CheckPrecision =
    (std::same_as<PlanType, fftwf_plan> and NumericConcepts::Float<Real>) or
    (std::same_as<PlanType, fftw_plan> and NumericConcepts::Double<Real>) or
    (std::same_as<PlanType, fftwl_plan> and NumericConcepts::LongDouble<Real>);

//--------------------------------------------------------------//
//                    Custom fftw3 allocator                    //
//--------------------------------------------------------------//

/**
 * @brief A custom STL allocator that uses `fftw_malloc` and `fftw_free`.
 * @details This ensures that memory allocated for containers like std::vector
 * is correctly aligned for SIMD instructions, as required by FFTW for optimal
 * performance.
 * @tparam T The type of the elements to be allocated.
 */
template <typename T>
class Allocator {
 public:
  using value_type = T;
  /** @brief Default constructor. */
  Allocator() noexcept {}
  /** @brief Copy constructor from an allocator of a different type. */
  template <class U>
  Allocator(const Allocator<U>&) noexcept {}
  /**
   * @brief Allocates `n` elements of type `T`.
   * @param n The number of elements to allocate.
   * @return A pointer to the allocated memory.
   */
  T* allocate(std::size_t n) {
    return static_cast<T*>(fftw_malloc(sizeof(T) * n));
  }
  /**
   * @brief Deallocates memory previously allocated with `allocate`.
   * @param p A pointer to the memory to deallocate.
   * @param n The number of elements that were allocated (unused).
   */
  void deallocate(T* p, std::size_t n) { fftw_free(p); }
};

/**
 * @brief Compares two allocators for equality.
 * @return Always returns true, as FFTW allocators are stateless.
 */
template <class T, class U>
constexpr bool operator==(const Allocator<T>&, const Allocator<U>&) noexcept {
  return true;
}

/**
 * @brief Compares two allocators for inequality.
 * @return Always returns false, as FFTW allocators are stateless.
 */
template <class T, class U>
constexpr bool operator!=(const Allocator<T>&, const Allocator<U>&) noexcept {
  return false;
}

/**
 * @brief Alias for `std::vector` using the custom FFTW-aligned allocator.
 * @tparam T The element type of the vector.
 */
template <typename T>
using vector = std::vector<T, Allocator<T>>;

/**
 * @brief Cleans up FFTW's process-global state for all supported precisions.
 * @details This function calls `fftwf_cleanup`, `fftw_cleanup`, and
 * `fftwl_cleanup`. It releases wisdom and other serial FFTW planner state, but
 * it does not destroy plans or clean up the optional FFTW threads interfaces.
 * All plans must be destroyed before this function is called.
 */
inline void CleanUp() {
  fftwf_cleanup();
  fftw_cleanup();
  fftwl_cleanup();
}

/**
 * @brief Safely casts a pointer to `std::complex<Real>` to the corresponding
 * FFTW complex type.
 * @tparam Real The floating-point precision (`float`, `double`, `long double`).
 * @param z A pointer to the `std::complex` data.
 * @return A pointer to the data cast as the appropriate FFTW complex type.
 */
template <NumericConcepts::Real Real>
auto ComplexCast(std::complex<Real>* z) {
  if constexpr (NumericConcepts::Float<Real>) {
    return reinterpret_cast<fftwf_complex*>(z);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return reinterpret_cast<fftw_complex*>(z);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return reinterpret_cast<fftwl_complex*>(z);
  }
}

//----------------------------------------------------------//
//                         1D plans                         //
//----------------------------------------------------------//

/**
 * @brief Creates a plan for a 1D complex-to-complex Discrete Fourier Transform
 * (DFT).
 * @tparam Real The floating-point precision of the data.
 * @param n The size of the transform.
 * @param in Pointer to the complex input array.
 * @param out Pointer to the complex output array.
 * @param sign The sign of the exponent in the DFT. Use `FFTW_FORWARD` (-1) or
 * `FFTW_BACKWARD` (+1).
 * @param flag A bitwise OR of FFTW planner flags (e.g., `FFTW_MEASURE`,
 * `FFTW_ESTIMATE`).
 * @return An FFTW plan handle corresponding to the data precision.
 */
template <NumericConcepts::Real Real>
auto Plan(int n, std::complex<Real>* in, std::complex<Real>* out, int sign,
          unsigned flag) {
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_dft_1d(n, ComplexCast(in), ComplexCast(out), sign, flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_dft_1d(n, ComplexCast(in), ComplexCast(out), sign, flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_dft_1d(n, ComplexCast(in), ComplexCast(out), sign, flag);
  }
}

/**
 * @brief Creates a plan for a 1D real-to-complex (R2C) DFT.
 * @tparam Real The floating-point precision of the data.
 * @param n The size of the transform.
 * @param in Pointer to the real input array.
 * @param out Pointer to the complex output array. The size should be n/2 + 1.
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle corresponding to the data precision.
 */
template <NumericConcepts::Real Real>
auto Plan(int n, Real* in, std::complex<Real>* out, unsigned flag) {
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_dft_r2c_1d(n, in, ComplexCast(out), flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_dft_r2c_1d(n, in, ComplexCast(out), flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_dft_r2c_1d(n, in, ComplexCast(out), flag);
  }
}

/**
 * @brief Creates a plan for a 1D complex-to-real (C2R) DFT.
 * @tparam Real The floating-point precision of the data.
 * @param n The size of the transform.
 * @param in Pointer to the complex input array. The size should be n/2 + 1.
 * @param out Pointer to the real output array.
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle corresponding to the data precision.
 */
template <NumericConcepts::Real Real>
auto Plan(int n, std::complex<Real>* in, Real* out, unsigned flag) {
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_dft_c2r_1d(n, ComplexCast(in), out, flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_dft_c2r_1d(n, ComplexCast(in), out, flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_dft_c2r_1d(n, ComplexCast(in), out, flag);
  }
}

/**
 * @brief Creates a plan for a 1D real-to-real (R2R) transform.
 * @tparam Real The floating-point precision of the data.
 * @param n The size of the transform.
 * @param in Pointer to the real input array.
 * @param out Pointer to the real output array.
 * @param kind The kind of R2R transform (e.g., `FFTW_REDFT00`, `FFTW_DHT`).
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle corresponding to the data precision.
 */
template <NumericConcepts::Real Real>
auto Plan(int n, Real* in, Real* out, fftw_r2r_kind kind, unsigned flag) {
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_r2r_1d(n, in, out, kind, flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_r2r_1d(n, in, out, kind, flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_r2r_1d(n, in, out, kind, flag);
  }
}

//----------------------------------------------------------//
//                         2D plans                         //
//----------------------------------------------------------//

/**
 * @brief Creates a plan for a 2D complex-to-complex DFT.
 * @tparam Real The floating-point precision of the data.
 * @param n0 The size of the transform in the first dimension.
 * @param n1 The size of the transform in the second dimension.
 * @param in Pointer to the complex input array.
 * @param out Pointer to the complex output array.
 * @param sign The sign of the exponent in the DFT (`FFTW_FORWARD` or
 * `FFTW_BACKWARD`).
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle.
 */
template <NumericConcepts::Real Real>
auto Plan(int n0, int n1, std::complex<Real>* in, std::complex<Real>* out,
          int sign, unsigned flag) {
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_dft_2d(n0, n1, ComplexCast(in), ComplexCast(out), sign,
                             flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_dft_2d(n0, n1, ComplexCast(in), ComplexCast(out), sign,
                            flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_dft_2d(n0, n1, ComplexCast(in), ComplexCast(out), sign,
                             flag);
  }
}

/**
 * @brief Creates a plan for a 2D real-to-complex (R2C) DFT.
 * @tparam Real The floating-point precision of the data.
 * @param n0 The size of the transform in the first dimension.
 * @param n1 The size of the transform in the second dimension.
 * @param in Pointer to the real input array.
 * @param out Pointer to the complex output array.
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle.
 */
template <NumericConcepts::Real Real>
auto Plan(int n0, int n1, Real* in, std::complex<Real>* out, unsigned flag) {
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_dft_r2c_2d(n0, n1, in, ComplexCast(out), flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_dft_r2c_2d(n0, n1, in, ComplexCast(out), flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_dft_r2c_2d(n0, n1, in, ComplexCast(out), flag);
  }
}

/**
 * @brief Creates a plan for a 2D complex-to-real (C2R) DFT.
 * @tparam Real The floating-point precision of the data.
 * @param n0 The size of the transform in the first dimension.
 * @param n1 The size of the transform in the second dimension.
 * @param in Pointer to the complex input array.
 * @param out Pointer to the real output array.
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle.
 */
template <NumericConcepts::Real Real>
auto Plan(int n0, int n1, std::complex<Real>* in, Real* out, unsigned flag) {
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_dft_c2r_2d(n0, n1, ComplexCast(in), out, flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_dft_c2r_2d(n0, n1, ComplexCast(in), out, flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_dft_c2r_2d(n0, n1, ComplexCast(in), out, flag);
  }
}

/**
 * @brief Creates a plan for a 2D real-to-real (R2R) transform.
 * @tparam Real The floating-point precision of the data.
 * @param n0 The size of the transform in the first dimension.
 * @param n1 The size of the transform in the second dimension.
 * @param in Pointer to the real input array.
 * @param out Pointer to the real output array.
 * @param kind0 The kind of R2R transform for the first dimension.
 * @param kind1 The kind of R2R transform for the second dimension.
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle.
 */
template <NumericConcepts::Real Real>
auto Plan(int n0, int n1, Real* in, Real* out, fftw_r2r_kind kind0,
          fftw_r2r_kind kind1, unsigned flag) {
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_r2r_2d(n0, n1, in, out, kind0, kind1, flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_r2r_2d(n0, n1, in, out, kind0, kind1, flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_r2r_2d(n0, n1, in, out, kind0, kind1, flag);
  }
}

//----------------------------------------------------------//
//                         3D plans                         //
//----------------------------------------------------------//

/**
 * @brief Creates a plan for a 3D complex-to-complex DFT.
 * @tparam Real The floating-point precision of the data.
 * @param n0 The size of the transform in the first dimension.
 * @param n1 The size of the transform in the second dimension.
 * @param n2 The size of the transform in the third dimension.
 * @param in Pointer to the complex input array.
 * @param out Pointer to the complex output array.
 * @param sign The sign of the exponent in the DFT (`FFTW_FORWARD` or
 * `FFTW_BACKWARD`).
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle.
 */
template <NumericConcepts::Real Real>
auto Plan(int n0, int n1, int n2, std::complex<Real>* in,
          std::complex<Real>* out, int sign, unsigned flag) {
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_dft_3d(n0, n1, n2, ComplexCast(in), ComplexCast(out),
                             sign, flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_dft_3d(n0, n1, n2, ComplexCast(in), ComplexCast(out), sign,
                            flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_dft_3d(n0, n1, n2, ComplexCast(in), ComplexCast(out),
                             sign, flag);
  }
}

/**
 * @brief Creates a plan for a 3D real-to-complex (R2C) DFT.
 * @tparam Real The floating-point precision of the data.
 * @param n0 The size of the transform in the first dimension.
 * @param n1 The size of the transform in the second dimension.
 * @param n2 The size of the transform in the third dimension.
 * @param in Pointer to the real input array.
 * @param out Pointer to the complex output array.
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle.
 */
template <NumericConcepts::Real Real>
auto Plan(int n0, int n1, int n2, Real* in, std::complex<Real>* out,
          unsigned flag) {
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_dft_r2c_3d(n0, n1, n2, in, ComplexCast(out), flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_dft_r2c_3d(n0, n1, n2, in, ComplexCast(out), flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_dft_r2c_3d(n0, n1, n2, in, ComplexCast(out), flag);
  }
}

/**
 * @brief Creates a plan for a 3D complex-to-real (C2R) DFT.
 * @tparam Real The floating-point precision of the data.
 * @param n0 The size of the transform in the first dimension.
 * @param n1 The size of the transform in the second dimension.
 * @param n2 The size of the transform in the third dimension.
 * @param in Pointer to the complex input array.
 * @param out Pointer to the real output array.
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle.
 */
template <NumericConcepts::Real Real>
auto Plan(int n0, int n1, int n2, std::complex<Real>* in, Real* out,
          unsigned flag) {
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_dft_c2r_3d(n0, n1, n2, ComplexCast(in), out, flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_dft_c2r_3d(n0, n1, n2, ComplexCast(in), out, flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_dft_c2r_3d(n0, n1, n2, ComplexCast(in), out, flag);
  }
}

/**
 * @brief Creates a plan for a 3D real-to-real (R2R) transform.
 * @tparam Real The floating-point precision of the data.
 * @param n0 The size of the transform in the first dimension.
 * @param n1 The size of the transform in the second dimension.
 * @param n2 The size of the transform in the third dimension.
 * @param in Pointer to the real input array.
 * @param out Pointer to the real output array.
 * @param kind0 The kind of R2R transform for the first dimension.
 * @param kind1 The kind of R2R transform for the second dimension.
 * @param kind2 The kind of R2R transform for the third dimension.
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle.
 */
template <NumericConcepts::Real Real>
auto Plan(int n0, int n1, int n2, Real* in, Real* out, fftw_r2r_kind kind0,
          fftw_r2r_kind kind1, fftw_r2r_kind kind2, unsigned flag) {
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_r2r_3d(n0, n1, n2, in, out, kind0, kind1, kind2, flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_r2r_3d(n0, n1, n2, in, out, kind0, kind1, kind2, flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_r2r_3d(n0, n1, n2, in, out, kind0, kind1, kind2, flag);
  }
}

//----------------------------------------------------------//
//                   Multi-dimensional plans                //
//----------------------------------------------------------//

/**
 * @brief Creates a plan for a multi-dimensional (rank > 0) complex-to-complex
 * DFT.
 * @tparam Real The floating-point precision of the data.
 * @param rank The number of dimensions for the transform.
 * @param n Pointer to an array of size `rank` specifying the dimensions.
 * @param in Pointer to the complex input array.
 * @param out Pointer to the complex output array.
 * @param sign The sign of the exponent (`FFTW_FORWARD` or `FFTW_BACKWARD`).
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle.
 */
template <NumericConcepts::Real Real>
auto Plan(int rank, int* n, std::complex<Real>* in, std::complex<Real>* out,
          int sign, unsigned flag) {
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_dft(rank, n, ComplexCast(in), ComplexCast(out), sign,
                          flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_dft(rank, n, ComplexCast(in), ComplexCast(out), sign,
                         flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_dft(rank, n, ComplexCast(in), ComplexCast(out), sign,
                          flag);
  }
}

/**
 * @brief Creates a plan for a multi-dimensional (rank > 0) real-to-complex DFT.
 * @tparam Real The floating-point precision of the data.
 * @param rank The number of dimensions.
 * @param n Pointer to an array of size `rank` specifying the dimensions.
 * @param in Pointer to the real input array.
 * @param out Pointer to the complex output array.
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle.
 */
template <NumericConcepts::Real Real>
auto Plan(int rank, int* n, Real* in, std::complex<Real>* out, unsigned flag) {
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_dft_r2c(rank, n, in, ComplexCast(out), flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_dft_r2c(rank, n, in, ComplexCast(out), flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_dft_r2c(rank, n, in, ComplexCast(out), flag);
  }
}

/**
 * @brief Creates a plan for a multi-dimensional (rank > 0) complex-to-real DFT.
 * @tparam Real The floating-point precision of the data.
 * @param rank The number of dimensions.
 * @param n Pointer to an array of size `rank` specifying the dimensions.
 * @param in Pointer to the complex input array.
 * @param out Pointer to the real output array.
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle.
 */
template <NumericConcepts::Real Real>
auto Plan(int rank, int* n, std::complex<Real>* in, Real* out, unsigned flag) {
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_dft_c2r(rank, n, ComplexCast(in), out, flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_dft_c2r(rank, n, ComplexCast(in), out, flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_dft_c2r(rank, n, ComplexCast(in), out, flag);
  }
}

/**
 * @brief Creates a plan for a multi-dimensional (rank > 0) real-to-real
 * transform.
 * @tparam Real The floating-point precision of the data.
 * @param rank The number of dimensions.
 * @param n Pointer to an array of size `rank` specifying the dimensions.
 * @param in Pointer to the real input array.
 * @param out Pointer to the real output array.
 * @param kind Pointer to an array of `fftw_r2r_kind` of size `rank`.
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle.
 */
template <NumericConcepts::Real Real>
auto Plan(int rank, int* n, Real* in, Real* out, fftw_r2r_kind* kind,
          unsigned flag) {
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_r2r(rank, n, in, out, kind, flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_r2r(rank, n, in, out, kind, flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_r2r(rank, n, in, out, kind, flag);
  }
}

//-------------------------------------------------------------//
//                      Advanced interface                     //
//-------------------------------------------------------------//

/**
 * @brief Creates a plan for multiple, strided, multi-dimensional
 * complex-to-complex DFTs.
 * @details This is a wrapper for `fftw_plan_many_dft` and its
 * precision-specific variants.
 * @tparam Real The floating-point precision of the data.
 * @param rank Number of dimensions.
 * @param n Array of dimensions.
 * @param howMany The number of transforms to compute.
 * @param in Pointer to the complex input data.
 * @param inEmbed The "embedded" dimensions of the input array (for sub-arrays).
 * Can be `nullptr`.
 * @param inStride The distance between consecutive elements in a dimension.
 * @param inDist The distance between the start of consecutive datasets.
 * @param out Pointer to the complex output data.
 * @param outEmbed The "embedded" dimensions of the output array. Can be
 * `nullptr`.
 * @param outStride The distance between consecutive elements in a dimension of
 * the output.
 * @param outDist The distance between the start of consecutive output datasets.
 * @param sign The sign of the exponent (`FFTW_FORWARD` or `FFTW_BACKWARD`).
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle.
 */
template <NumericConcepts::Real Real>
auto Plan(int rank, int* n, int howMany, std::complex<Real>* in, int* inEmbed,
          int inStride, int inDist, std::complex<Real>* out, int* outEmbed,
          int outStride, int outDist, int sign, unsigned flag) {
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_many_dft(rank, n, howMany, ComplexCast(in), inEmbed,
                               inStride, inDist, ComplexCast(out), outEmbed,
                               outStride, outDist, sign, flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_many_dft(rank, n, howMany, ComplexCast(in), inEmbed,
                              inStride, inDist, ComplexCast(out), outEmbed,
                              outStride, outDist, sign, flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_many_dft(rank, n, howMany, ComplexCast(in), inEmbed,
                               inStride, inDist, ComplexCast(out), outEmbed,
                               outStride, outDist, sign, flag);
  }
}

/**
 * @brief Creates a plan for multiple, strided, multi-dimensional
 * real-to-complex DFTs.
 * @tparam Real The floating-point precision of the data.
 * @param rank Number of dimensions.
 * @param n Array of dimensions.
 * @param howMany The number of transforms to compute.
 * @param in Pointer to the real input data.
 * @param inEmbed The "embedded" dimensions of the input array. Can be
 * `nullptr`.
 * @param inStride The distance between consecutive elements in a dimension.
 * @param inDist The distance between the start of consecutive datasets.
 * @param out Pointer to the complex output data.
 * @param outEmbed The "embedded" dimensions of the output array. Can be
 * `nullptr`.
 * @param outStride The distance between consecutive elements in a dimension of
 * the output.
 * @param outDist The distance between the start of consecutive output datasets.
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle.
 */
template <NumericConcepts::Real Real>
auto Plan(int rank, int* n, int howMany, Real* in, int* inEmbed, int inStride,
          int inDist, std::complex<Real>* out, int* outEmbed, int outStride,
          int outDist, unsigned flag) {
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_many_dft_r2c(rank, n, howMany, in, inEmbed, inStride,
                                   inDist, ComplexCast(out), outEmbed,
                                   outStride, outDist, flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_many_dft_r2c(rank, n, howMany, in, inEmbed, inStride,
                                  inDist, ComplexCast(out), outEmbed, outStride,
                                  outDist, flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_many_dft_r2c(rank, n, howMany, in, inEmbed, inStride,
                                   inDist, ComplexCast(out), outEmbed,
                                   outStride, outDist, flag);
  }
}

/**
 * @brief Creates a plan for multiple, strided, multi-dimensional
 * complex-to-real DFTs.
 * @tparam Real The floating-point precision of the data.
 * @param rank Number of dimensions.
 * @param n Array of dimensions.
 * @param howMany The number of transforms to compute.
 * @param in Pointer to the complex input data.
 * @param inEmbed The "embedded" dimensions of the input array. Can be
 * `nullptr`.
 * @param inStride The distance between consecutive elements in a dimension.
 * @param inDist The distance between the start of consecutive datasets.
 * @param out Pointer to the real output data.
 * @param outEmbed The "embedded" dimensions of the output array. Can be
 * `nullptr`.
 * @param outStride The distance between consecutive elements in a dimension of
 * the output.
 * @param outDist The distance between the start of consecutive output datasets.
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle.
 */
template <NumericConcepts::Real Real>
auto Plan(int rank, int* n, int howMany, std::complex<Real>* in, int* inEmbed,
          int inStride, int inDist, Real* out, int* outEmbed, int outStride,
          int outDist, unsigned flag) {
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_many_dft_c2r(rank, n, howMany, ComplexCast(in), inEmbed,
                                   inStride, inDist, out, outEmbed, outStride,
                                   outDist, flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_many_dft_c2r(rank, n, howMany, ComplexCast(in), inEmbed,
                                  inStride, inDist, out, outEmbed, outStride,
                                  outDist, flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_many_dft_c2r(rank, n, howMany, ComplexCast(in), inEmbed,
                                   inStride, inDist, out, outEmbed, outStride,
                                   outDist, flag);
  }
}

/**
 * @brief Creates a plan for multiple, strided, multi-dimensional real-to-real
 * transforms.
 * @tparam Real The floating-point precision of the data.
 * @param rank Number of dimensions.
 * @param n Array of dimensions.
 * @param howMany The number of transforms to compute.
 * @param in Pointer to the real input data.
 * @param inEmbed The "embedded" dimensions of the input array. Can be
 * `nullptr`.
 * @param inStride The distance between consecutive elements in a dimension.
 * @param inDist The distance between the start of consecutive datasets.
 * @param out Pointer to the real output data.
 * @param outEmbed The "embedded" dimensions of the output array. Can be
 * `nullptr`.
 * @param outStride The distance between consecutive elements in a dimension of
 * the output.
 * @param outDist The distance between the start of consecutive output datasets.
 * @param kind Pointer to an array of `fftw_r2r_kind` of size `rank`.
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle.
 */
template <NumericConcepts::Real Real>
auto Plan(int rank, int* n, int howMany, Real* in, int* inEmbed, int inStride,
          int inDist, Real* out, int* outEmbed, int outStride, int outDist,
          fftw_r2r_kind* kind, unsigned flag) {
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_many_r2r(rank, n, howMany, in, inEmbed, inStride, inDist,
                               out, outEmbed, outStride, outDist, kind, flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_many_r2r(rank, n, howMany, in, inEmbed, inStride, inDist,
                              out, outEmbed, outStride, outDist, kind, flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_many_r2r(rank, n, howMany, in, inEmbed, inStride, inDist,
                               out, outEmbed, outStride, outDist, kind, flag);
  }
}

//----------------------------------------------------------//
//                 Plan destruction functions               //
//----------------------------------------------------------//

/**
 * @brief Destroys a given FFTW plan and frees associated resources.
 * @tparam PlanType The type of the plan handle, constrained by `IsPlan`.
 * @param plan The plan to destroy. Must not be null.
 */
template <IsPlan PlanType>
void Destroy(PlanType plan) {
  assert(plan != nullptr);
  if constexpr (std::same_as<PlanType, fftwf_plan>) {
    fftwf_destroy_plan(plan);
  }
  if constexpr (std::same_as<PlanType, fftw_plan>) {
    fftw_destroy_plan(plan);
  }
  if constexpr (std::same_as<PlanType, fftwl_plan>) {
    fftwl_destroy_plan(plan);
  }
}

//----------------------------------------------------------//
//                  Plan execution functions                //
//----------------------------------------------------------//

/**
 * @brief Executes a plan that was created for in-place transforms.
 * @details This is for plans where the input and output pointers were the same
 * during creation. It wraps the appropriate `fftw*_execute` function based on
 * the plan's precision.
 * @tparam PlanType The type of the plan handle, constrained by `IsPlan`.
 * @param plan The plan to execute. Must not be null.
 */
template <IsPlan PlanType>
void Execute(PlanType plan) {
  assert(plan != nullptr);
  if constexpr (std::same_as<PlanType, fftwf_plan>) {
    fftwf_execute(plan);
  }
  if constexpr (std::same_as<PlanType, fftw_plan>) {
    fftw_execute(plan);
  }
  if constexpr (std::same_as<PlanType, fftwl_plan>) {
    fftwl_execute(plan);
  }
}

/**
 * @brief Executes an out-of-place complex-to-complex DFT plan with new arrays.
 * @details Wraps the appropriate `fftw*_execute_dft` function based on the
 * plan's precision. This allows reusing a plan with different input/output
 * buffers that have the same alignment properties as the ones used for
 * planning.
 * @tparam PlanType The type of the plan handle.
 * @tparam Real The floating-point precision of the data.
 * @param plan The plan to execute.
 * @param in Pointer to the complex input array.
 * @param out Pointer to the complex output array.
 */
template <typename PlanType, NumericConcepts::Real Real>
requires CheckPrecision<PlanType, Real>
void Execute(PlanType plan, std::complex<Real>* in, std::complex<Real>* out) {
  if constexpr (std::same_as<PlanType, fftwf_plan>) {
    fftwf_execute_dft(plan, ComplexCast(in), ComplexCast(out));
  }
  if constexpr (std::same_as<PlanType, fftw_plan>) {
    fftw_execute_dft(plan, ComplexCast(in), ComplexCast(out));
  }
  if constexpr (std::same_as<PlanType, fftwl_plan>) {
    fftwl_execute_dft(plan, ComplexCast(in), ComplexCast(out));
  }
}

/**
 * @brief Executes an out-of-place real-to-complex DFT plan with new arrays.
 * @details Wraps the appropriate `fftw*_execute_dft_r2c` function based on the
 * plan's precision.
 * @tparam PlanType The type of the plan handle.
 * @tparam Real The floating-point precision of the data.
 * @param plan The plan to execute.
 * @param in Pointer to the real input array.
 * @param out Pointer to the complex output array.
 */
template <typename PlanType, NumericConcepts::Real Real>
requires CheckPrecision<PlanType, Real>
void Execute(PlanType plan, Real* in, std::complex<Real>* out) {
  if constexpr (std::same_as<PlanType, fftwf_plan>) {
    fftwf_execute_dft_r2c(plan, in, ComplexCast(out));
  }
  if constexpr (std::same_as<PlanType, fftw_plan>) {
    fftw_execute_dft_r2c(plan, in, ComplexCast(out));
  }
  if constexpr (std::same_as<PlanType, fftwl_plan>) {
    fftwl_execute_dft_r2c(plan, in, ComplexCast(out));
  }
}

/**
 * @brief Executes an out-of-place complex-to-real DFT plan with new arrays.
 * @details Wraps the appropriate `fftw*_execute_dft_c2r` function based on the
 * plan's precision.
 * @tparam PlanType The type of the plan handle.
 * @tparam Real The floating-point precision of the data.
 * @param plan The plan to execute.
 * @param in Pointer to the complex input array.
 * @param out Pointer to the real output array.
 */
template <typename PlanType, NumericConcepts::Real Real>
requires CheckPrecision<PlanType, Real>
void Execute(PlanType plan, std::complex<Real>* in, Real* out) {
  if constexpr (std::same_as<PlanType, fftwf_plan>) {
    fftwf_execute_dft_c2r(plan, ComplexCast(in), out);
  }
  if constexpr (std::same_as<PlanType, fftw_plan>) {
    fftw_execute_dft_c2r(plan, ComplexCast(in), out);
  }
  if constexpr (std::same_as<PlanType, fftwl_plan>) {
    fftwl_execute_dft_c2r(plan, ComplexCast(in), out);
  }
}

/**
 * @brief Executes an out-of-place real-to-real transform plan with new arrays.
 * @details Wraps the appropriate `fftw*_execute_r2r` function based on the
 * plan's precision.
 * @tparam PlanType The type of the plan handle.
 * @tparam Real The floating-point precision of the data.
 * @param plan The plan to execute.
 * @param in Pointer to the real input array.
 * @param out Pointer to the real output array.
 */
template <typename PlanType, NumericConcepts::Real Real>
requires CheckPrecision<PlanType, Real>
void Execute(PlanType plan, Real* in, Real* out) {
  if constexpr (std::same_as<PlanType, fftwf_plan>) {
    fftwf_execute_r2r(plan, in, out);
  }
  if constexpr (std::same_as<PlanType, fftw_plan>) {
    fftw_execute_r2r(plan, in, out);
  }
  if constexpr (std::same_as<PlanType, fftwl_plan>) {
    fftwl_execute_r2r(plan, in, out);
  }
}

}  // namespace FFTWpp

#endif  // FFTWPP_CORE_GUARD_H
