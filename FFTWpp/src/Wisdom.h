/**
 * @file Wisdom.h
 * @brief Provides utility functions for managing FFTW wisdom.
 *
 * Wisdom in FFTW is a way to save and reuse information about how to compute
 * transforms of a given size and type optimally. This can significantly speed
 * up the creation of new plans. This file provides functions to import, export,
 * forget, and pre-generate wisdom for various transform types.
 */
#ifndef FFTWPP_WISDOM_GUARD_H
#define FFTWPP_WISDOM_GUARD_H

#include <cassert>
#include <string>

#include "NumericConcepts/Numeric.hpp"
#include "NumericConcepts/Ranges.hpp"
#include "Options.h"
#include "Plan.h"
#include "Views.h"
#include "fftw3.h"

namespace FFTWpp {

/**
 * @brief Exports all accumulated wisdom to a file.
 * @details This function calls `fftw_export_wisdom_to_filename`, which saves
 * wisdom for all precisions (float, double, and long double) that FFTW knows
 * about. The function will assert if the file cannot be written to.
 * @param filename The path to the file where wisdom will be saved.
 */
void ExportWisdom(const std::string& filename) {
  int io = fftw_export_wisdom_to_filename(filename.c_str());
  assert(io == 0);
}

/**
 * @brief Imports wisdom from a file.
 * @details This function calls `fftw_import_wisdom_from_filename`, which reads
 * wisdom from a file and merges it with any existing wisdom in memory. This can
 * be called multiple times. The function will assert if the file cannot be
 * read.
 * @param filename The path to the file from which to load wisdom.
 */
void ImportWisdom(const std::string& filename) {
  int io = fftw_import_wisdom_from_filename(filename.c_str());
  assert(io == 0);
}

/**
 * @brief Forgets all accumulated double-precision wisdom.
 * @warning This function only calls `fftw_forget_wisdom()`, which affects
 * double-precision plans. It does NOT forget wisdom for float (`fftwf_plan`)
 * or long double (`fftwl_plan`) precisions. To forget all wisdom, one
 * must call `fftwf_forget_wisdom()` and `fftwl_forget_wisdom()` separately.
 */
void ForgetWisdom() { fftw_forget_wisdom(); }

/**
 * @brief Generates wisdom for complex-to-complex, real-to-complex, or
 * complex-to-real transforms.
 * @details This function creates temporary data buffers and then creates (but
 * does not execute) forward and backward plans for the specified layouts. This
 * process populates the internal FFTW wisdom cache for the given transform
 * size, type, and flags. No wisdom is generated if the flag is `Estimate`.
 * @tparam InType The value type of the input data (e.g.,
 * `std::complex<double>`).
 * @tparam OutType The value type of the output data (e.g., `double`).
 * @param inLayout The layout (shape) of the input data.
 * @param outLayout The layout (shape) of the output data.
 * @param flag The planner flag (`Measure`, `Patient`, etc.) to use for wisdom
 * generation.
 * @requires The precision of InType and OutType must be the same.
 */
template <NumericConcepts::RealOrComplex InType,
          NumericConcepts::RealOrComplex OutType>
requires NumericConcepts::SamePrecision<InType, OutType>
void GenerateWisdom(Ranges::Layout inLayout, Ranges::Layout outLayout,
                    Flag flag) {
  if (flag == Estimate) return;
  auto in = vector<InType>(inLayout.size());
  auto inView = Ranges::View(in, inLayout);
  auto out = vector<OutType>(outLayout.size());
  auto outView = Ranges::View(out, outLayout);
  if constexpr (NumericConcepts::Complex<InType> &&
                NumericConcepts::Complex<OutType>) {
    auto plan = Ranges::Plan(inView, outView, flag, Forward);
    plan = Ranges::Plan(outView, inView, flag, Backward);
  }
  if constexpr ((NumericConcepts::Real<InType> &&
                 NumericConcepts::Complex<OutType>) ||
                (NumericConcepts::Complex<InType> &&
                 NumericConcepts::Real<OutType>)) {
    auto planForward = Ranges::Plan(inView, outView, flag);
    auto planBackward = Ranges::Plan(outView, inView, flag);
  }
}

/**
 * @brief Generates wisdom for real-to-real transforms.
 * @details This function populates the FFTW wisdom cache for R2R transforms of
 * a given layout and kind. It creates plans for both the forward and backward
 * transforms.
 * @warning The current implementation for generating the backward plan's kinds
 * creates a lazy `std::ranges::views::transform` whose result is discarded.
 * This means the backward plan is likely created with the original forward
 * kinds, not their inverses as was probably intended.
 * @tparam InType Real input value type.
 * @tparam OutType Real output value type.
 * @param inLayout The layout (shape) of the input data.
 * @param outLayout The layout (shape) of the output data.
 * @param kinds An initializer list of `RealKind` for the forward transform.
 * @param flag The planner flag (`Measure`, `Patient`, etc.) to use.
 * @requires InType and OutType must be real types with the same precision.
 */
template <NumericConcepts::Real InType, NumericConcepts::Real OutType>
requires NumericConcepts::SamePrecision<InType, OutType>
void GenerateWisdom(Ranges::Layout inLayout, Ranges::Layout outLayout,
                    std::initializer_list<RealKind> kinds, Flag flag) {
  if (flag == Estimate) return;
  auto in = vector<InType>(inLayout.size());
  auto inView = Ranges::View(in, inLayout);
  auto out = vector<OutType>(outLayout.size());
  auto outView = Ranges::View(out, outLayout);
  auto planForward = Ranges::Plan(inView, outView, kinds, flag);
  auto kindsBackward = kinds;
  std::ranges::views::all(kindsBackward) |
      std::ranges::views::transform([](auto kind) { return kind.Inverse(); });
  auto planBackward = Ranges::Plan(outView, inView, kindsBackward, flag);
}

}  // namespace FFTWpp

#endif  // FFTWPP_WISDOM_GUARD_H