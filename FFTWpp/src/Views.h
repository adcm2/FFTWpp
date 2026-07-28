/**
 * @file Views.h
 * @brief Provides C++20-style views and layout descriptors for FFTW data.
 *
 * This file defines two main classes: `Layout` and `View`. The `Layout` class
 * describes the N-dimensional shape and memory layout of data for an FFTW
 * transform, corresponding to the parameters of the advanced FFTW interface.
 * The `View` class combines a `Layout` with a C++20 range, providing a unified
 * object that can be used with the `FFTWpp::Ranges::Plan` class.
 */
#ifndef FFTWPP_VIEWS_GUARD_H
#define FFTWPP_VIEWS_GUARD_H

#include <algorithm>
#include <cassert>
#include <complex>
#include <concepts>
#include <memory>
#include <numeric>
#include <ranges>
#include <vector>

#include "Core.h"
#include "NumericConcepts/Numeric.hpp"
#include "NumericConcepts/Ranges.hpp"
#include "fftw3.h"

namespace FFTWpp {

namespace Ranges {

/**
 * @class Layout
 * @brief Describes the memory layout and dimensions of an FFTW transform.
 * @details This class holds all the parameters required by the advanced FFTW
 * planning routines, such as rank, dimensions, stride, and distance. It defines
 * the "shape" of the transform data.
 */
class Layout {
 public:
  /** @brief Default constructor. */
  Layout() = default;

  /**
   * @brief Constructor for simple, contiguous, multi-dimensional transforms.
   * @details Creates a layout for a single transform (`howMany`=1) with default
   * stride and distance.
   * @tparam Dimensions A parameter pack of integral types.
   * @param dimensions The size of the transform along each dimension.
   */
  template <typename... Dimensions>
  requires(sizeof...(Dimensions) > 0) and (std::integral<Dimensions> && ...)
  Layout(Dimensions... dimensions)
      : Layout(sizeof...(Dimensions), std::vector{dimensions...}, 1,
               std::vector{dimensions...}, 1, 0) {}

  /**
   * @brief General constructor for the advanced FFTW interface.
   * @tparam R1 A range type whose values are convertible to int.
   * @tparam R2 A range type whose values are convertible to int.
   * @param rank The number of dimensions of the transform.
   * @param n A range specifying the size of each dimension.
   * @param howMany The number of transforms to perform.
   * @param embed A range specifying the "embedded" size of the data arrays.
   * @param stride The distance between consecutive elements in a dimension.
   * @param dist The distance between the start of consecutive transform data
   * sets.
   */
  template <std::ranges::range R1, std::ranges::range R2>
  requires requires() {
    requires std::convertible_to<std::ranges::range_value_t<R1>, int>;
    requires std::convertible_to<std::ranges::range_value_t<R2>, int>;
  }
  Layout(int rank, R1&& n, int howMany, R2&& embed, int stride, int dist)
      : _rank{rank},
        _n{std::vector<int>(std::begin(n), std::end(n))},
        _howMany{howMany},
        _embed{std::vector<int>(std::begin(embed), std::end(embed))},
        _stride{stride},
        _dist{dist} {}

  // Access the layout information.
  /** @brief Gets the rank (number of dimensions) of the transform. */
  auto Rank() const { return _rank; }
  /** @brief Gets a view of the logical dimensions of the transform. */
  auto N() const { return std::views::all(_n); }
  /** @brief Gets the number of transforms to be computed. */
  auto HowMany() const { return _howMany; }
  /** @brief Gets a view of the embedded dimensions of the data array. */
  auto Embed() const { return std::views::all(_embed); }
  /** @brief Gets the stride between elements. */
  auto Stride() const { return _stride; }
  /** @brief Gets the distance between consecutive transforms. */
  auto Dist() const { return _dist; }

  // Return pointers to the storage vectors.
  /** @brief Gets a raw pointer to the logical dimensions vector (`n`). */
  auto NPointer() { return _n.data(); }
  /** @brief Gets a raw pointer to the embedded dimensions vector. */
  auto EmbedPointer() { return _embed.data(); }

  /**
   * @brief Calculates the total storage size required for this layout.
   * @return The total number of elements in memory.
   */
  auto size() const {
    return HowMany() * std::accumulate(Embed().begin(), Embed().end(), 1,
                                       std::multiplies<>());
  }

  /** @brief Defaulted equality operator. */
  bool operator==(const Layout&) const = default;

 private:
  /// @brief Rank of the transformations (i.e., 1D, 2D, etc).
  int _rank;
  /// @brief Vector of logical dimensions along each rank.
  std::vector<int> _n;
  /// @brief Number of transforms to be performed.
  int _howMany;
  /// @brief Embedded size of the array along each rank.
  std::vector<int> _embed;
  /// @brief Offset between elements of the data.
  int _stride;
  /// @brief Offset between the start of each transformation.
  int _dist;
};

/**
 * @class View
 * @brief A C++20 ranges view combined with an FFTW data layout.
 * @details This class adapts a C++20 view (or range) to be used with the
 * `FFTWpp::Ranges::Plan`. It inherits from `Layout` to provide the necessary
 * dimensional information to FFTW's planning routines, and from
 * `std::ranges::view_interface` to provide standard range-based access.
 * @tparam _View The underlying C++20 view type. Must satisfy
 * `RealOrComplexWritableView`.
 */
template <NumericConcepts::RealOrComplexWritableView _View>
class View : public std::ranges::view_interface<View<_View>>, public Layout {
  using std::ranges::view_interface<View<_View>>::size;

 public:
  /**
   * @brief Constructs a View from an existing view and a Layout object.
   * @param view The underlying data view.
   * @param layout The layout describing the transform shape.
   */
  View(_View view, Layout layout) : Layout(layout), _view{view} {
    assert(CheckSize());
  }

  /**
   * @brief Constructs a 1D View from an existing view.
   * @details The layout is inferred as a simple 1D transform of the view's
   * size.
   * @param view The underlying data view.
   */
  View(_View view) : View(view, Layout(view.size())) {}

  /**
   * @brief Constructs a multi-dimensional View from a view and dimensions.
   * @details Creates the `Layout` internally from the provided dimensions.
   * @tparam Dimensions A parameter pack of integral types for the dimensions.
   * @param view The underlying data view.
   * @param dimensions The size of the transform along each dimension.
   */
  template <typename... Dimensions>
  requires(sizeof...(Dimensions) > 0) and
          (std::convertible_to<Dimensions, int> && ...)
  View(_View view, Dimensions... dimensions)
      : View(view, Layout(dimensions...)) {
    assert(CheckSize());
  }

  // Methods to inherit from view_interface.
  /** @brief Returns an iterator to the beginning of the view. */
  auto begin() { return _view.begin(); }
  /** @brief Returns an iterator to the end of the view. */
  auto end() { return _view.end(); }

  /**
   * @brief Returns a raw pointer to the beginning of the underlying data.
   * @details This is required for interfacing with the FFTW C API.
   * @return A pointer to the first element.
   */
  auto DataPointer() { return _view.data(); }

 private:
  /// @brief The stored C++20 view to the data.
  _View _view;

  /**
   * @brief Checks that the underlying view size matches the required storage
   * size of the Layout.
   */
  auto CheckSize() const { return _view.size() == Layout::size(); }
};

/**
 * @brief Deduction guide to allow constructing a `View` from a range.
 * @details This allows a user to pass a container like `std::vector` directly
 * when constructing a `View`, automatically converting it to a view type.
 */
template <std::ranges::range R, typename... Args>
View(R&&, Args...) -> View<std::ranges::views::all_t<R>>;

}  // namespace Ranges

}  // namespace FFTWpp

#endif  // FFTWPP_VIEWS_GUARD_H
