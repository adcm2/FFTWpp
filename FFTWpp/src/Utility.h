/**
 * @file Utility.h
 * @brief Provides miscellaneous utility functions for testing and data
 * handling.
 *
 * This file contains helper functions for common tasks such as calculating
 * the required sizes of input and output arrays, filling ranges with random
 * data for testing, and verifying the results of transforms.
 */
#ifndef FFTWPP_UTILITY_GUARD_H
#define FFTWPP_UTILITY_GUARD_H

#include <algorithm>
#include <complex>
#include <random>
#include <ranges>

#include "Core.h"
#include "NumericConcepts/Numeric.hpp"
#include "NumericConcepts/Ranges.hpp"

namespace FFTWpp {

/**
 * @brief Calculates the required input and output array sizes for a transform.
 * @details For complex-to-complex transforms, input and output sizes are
 * identical. For real-to-complex or complex-to-real, one size is for the full
 * real data and the other is for the half-complex data (where the last
 * dimension is size N/2 + 1).
 * @tparam InType The value type of the input data.
 * @tparam OutType The value type of the output data.
 * @tparam Dimensions A parameter pack of integral types for the dimensions.
 * @param dimensions The size of the transform along each dimension.
 * @return A `std::pair` where `first` is the required input array size and
 * `second` is the required output array size.
 */
template <NumericConcepts::RealOrComplex InType,
          NumericConcepts::RealOrComplex OutType, typename... Dimensions>
requires(sizeof...(Dimensions) > 0) and (std::integral<Dimensions> && ...)
auto DataSize(Dimensions... dimensions) {
  auto dims = std::vector{{dimensions...}};
  auto size0 = std::ranges::fold_left_first(std::ranges::views::all(dims),
                                            std::multiplies<>())
                   .value();
  if constexpr (std::same_as<InType, OutType>) {
    return std::pair(size0, size0);
  } else {
    auto rank = dims.size();
    auto last = dims.back();
    auto size1 =
        std::ranges::fold_left_first(
            std::ranges::views::all(dims) | std::ranges::views::take(rank - 1),
            std::multiplies<>())
            .value_or(1) *
        (last / 2 + 1);
    if constexpr (NumericConcepts::Real<InType> &&
                  NumericConcepts::Complex<OutType>) {
      return std::pair(size0, size1);
    } else {
      return std::pair(size1, size0);
    }
  }
}

/**
 * @brief Fills a range with random values from a standard normal distribution.
 * @details Values are generated with a mean of 0.0 and a standard deviation
 * of 1.0. If the range contains complex numbers, both the real and imaginary
 * parts are filled with independent random values.
 * @tparam Range The type of the range to modify. Must be a writable range of
 * real or complex numbers.
 * @param range The range to fill with random values.
 */
template <NumericConcepts::RealOrComplexWritableRange Range>
void RandomiseValues(Range& range) {
  using Scalar = std::ranges::range_value_t<Range>;
  using Real = NumericConcepts::RemoveComplex<Scalar>;
  std::random_device rd{};
  std::mt19937_64 gen{rd()};
  std::normal_distribution<Real> d{0., 1.};
  std::transform(range.begin(), range.end(), range.begin(), [&](auto) {
    if constexpr (NumericConcepts::Real<Scalar>) {
      return d(gen);
    } else {
      return Scalar{d(gen), d(gen)};
    }
  });
}

/**
 * @brief Checks if two ranges are approximately equal after scaling one by a
 * norm.
 * @details This is useful for verifying the correctness of a
 * forward-and-backward transform. It computes `abs(in - copy * norm)` for each
 * element and checks if the result is within a tolerance defined by the machine
 * epsilon of the data type.
 * @tparam Range A readable range of real or complex numbers.
 * @tparam Scalar A numeric type for the normalization factor.
 * @param in The first range (e.g., the original data).
 * @param copy The second range (e.g., the result of the inverse transform).
 * @param norm The normalization factor to apply to the second range.
 * @return `true` if all corresponding values are approximately equal, `false`
 * otherwise.
 */
template <NumericConcepts::RealOrComplexRange Range, typename Scalar>
requires requires() {
  requires std::convertible_to<Scalar, std::ranges::range_value_t<Range>>;
}
auto CheckValues(Range&& in, Range&& copy, Scalar norm) {
  using Real = NumericConcepts::RemoveComplex<Scalar>;
  return std::ranges::all_of(
      std::ranges::views::zip_transform(
          [norm](auto x, auto y) { return std::abs(x - y * norm); },
          std::ranges::views::all(in), std::ranges::views::all(copy)),
      [](auto x) { return x < 1000 * std::numeric_limits<Real>::epsilon(); });
}

}  // namespace FFTWpp

#endif  // FFTWPP_UTILITY_GUARD_H
