/**
 * @file Plan.h
 * @brief Provides a high-level, RAII-compliant, range-based C++ wrapper for
 * FFTW plans.
 *
 * This file defines the `FFTWpp::Ranges::Plan` class, which encapsulates an
 * FFTW plan. It leverages C++20 ranges and views to provide a modern, safe, and
 * expressive interface for creating, managing, and executing FFTW transforms.
 * The `Plan` class handles resource management automatically via RAII.
 */
#ifndef FFTWPP_PLAN_GUARD_H
#define FFTWPP_PLAN_GUARD_H

#include <algorithm>
#include <cassert>
#include <complex>
#include <initializer_list>
#include <ranges>
#include <variant>

#include "Core.h"
#include "NumericConcepts/Numeric.hpp"
#include "NumericConcepts/Ranges.hpp"
#include "Options.h"
#include "Views.h"
#include "fftw3.h"

namespace FFTWpp {

/**
 * @brief Contains range-based wrappers for FFTW functionality.
 */
namespace Ranges {

/**
 * @class Plan
 * @brief A high-level, RAII-compliant wrapper for an FFTW plan using C++20
 * ranges.
 * @details This class manages the lifecycle of an FFTW plan. It is constructed
 * with `FFTWpp::View` objects that describe the input and output data layouts.
 * The appropriate FFTW plan is created based on the data types and transform
 * parameters. The plan is automatically destroyed when the `Plan` object goes
 * out of scope.
 *
 * @tparam InView The type of the input view, must satisfy
 * `RealOrComplexWritableRange`.
 * @tparam OutView The type of the output view, must satisfy
 * `RealOrComplexWritableRange`.
 * @requires The precision of the input and output views must be the same.
 */
template <NumericConcepts::RealOrComplexWritableRange InView,
          NumericConcepts::RealOrComplexWritableRange OutView>
requires NumericConcepts::SameRangePrecision<InView, OutView>

class Plan {
  /// @brief The value type of the input range (e.g., `std::complex<double>`).
  using InType = std::ranges::range_value_t<InView>;
  /// @brief The value type of the output range (e.g., `double`).
  using OutType = std::ranges::range_value_t<OutView>;
  /// @brief The underlying real precision of the data (e.g., `double`).
  using Real = NumericConcepts::RemoveComplex<InType>;

 public:
  /** @brief Default constructor is deleted. A plan must be initialized with
   * views. */
  Plan() = delete;

  /**
   * @brief Constructor for complex-to-complex (C2C) transforms.
   * @param in The input data view.
   * @param out The output data view.
   * @param flag The planner flag (`Estimate`, `Measure`, etc.).
   * @param direction The direction of the transform (`Forward` or `Backward`).
   * @requires Both InType and OutType must be complex types.
   */
  Plan(View<InView> in, View<OutView> out, Flag flag, Direction direction)
  requires NumericConcepts::Complex<InType> and
               NumericConcepts::Complex<OutType>
      : _in{in}, _out{out}, _flag{flag}, _direction{direction} {
    assert(CheckInputs());
    MakePlan(_flag);
  }

  /**
   * @brief Constructor for real-to-complex (R2C) or complex-to-real (C2R)
   * transforms.
   * @param in The input data view.
   * @param out The output data view.
   * @param flag The planner flag (`Estimate`, `Measure`, etc.).
   * @requires One of InType/OutType must be real and the other complex.
   */
  Plan(View<InView> in, View<OutView> out, Flag flag)
  requires(NumericConcepts::Complex<InType> and
           NumericConcepts::Real<OutType>) or
              (NumericConcepts::Real<InType> and
               NumericConcepts::Complex<OutType>)
      : _in{in}, _out{out}, _flag{flag} {
    assert(CheckInputs());
    MakePlan(_flag);
  }

  /**
   * @brief Constructor for real-to-real (R2R) transforms.
   * @details If fewer `kinds` are provided than the rank of the transform, the
   * last provided kind is used for all remaining dimensions.
   * @tparam RealKinds A parameter pack of `FFTWpp::RealKind`.
   * @param in The input data view.
   * @param out The output data view.
   * @param flag The planner flag (`Estimate`, `Measure`, etc.).
   * @param kinds A list of `RealKind` for each dimension of the transform.
   * @requires Both InType and OutType must be real types. At least one kind
   * must be specified.
   */
  template <typename... RealKinds>
  requires(sizeof...(RealKinds) > 0) and
              (std::same_as<RealKinds, RealKind> && ...)
  Plan(View<InView> in, View<OutView> out, Flag flag, RealKinds... kinds)
      : _in{in},
        _out{out},
        _flag{flag},
        _kinds{std::vector<RealKind>{kinds...}} {
    assert(Kinds().size() <= _in.Rank());
    if (Kinds().size() < _in.Rank()) {
      auto kinds = std::get<std::vector<RealKind>>(_kinds);
      while (kinds.size() < _in.Rank()) {
        kinds.push_back(kinds.back());
      }
      _kinds = kinds;
    }
    assert(CheckInputs());
    MakePlan(_flag);
  }

  /**
   * @brief Copy constructor. Creates a new plan based on the other's
   * configuration.
   * @details This creates a new FFTW plan. If the original plan was created
   * with `Measure`, this copy will be created with `WisdomOnly` to reuse the
   * wisdom, otherwise `Estimate` is used.
   * @param other The Plan object to copy from.
   */
  Plan(const Plan& other)
      : _in{other._in},
        _out{other._out},
        _flag{other._flag},
        _direction{other._direction},
        _kinds{other._kinds} {
    auto flag = _flag == Estimate ? Estimate : WisdomOnly;
    MakePlan(flag);
  }

  /**
   * @brief Move constructor. Takes ownership of the other plan's configuration.
   * @details The moved-from object's plan is destroyed. The new plan is created
   * using wisdom if available.
   * @param other The Plan object to move from.
   */
  Plan(Plan&& other)
      : _in{std::move(other._in)},
        _out{std::move(other._out)},
        _flag{std::move(other._flag)},
        _direction{std::move(other._direction)},
        _kinds{std::move(other._kinds)} {
    other.Destroy();
    auto flag = _flag == Estimate ? Estimate : WisdomOnly;
    MakePlan(flag);
  }

  /**
   * @brief Copy assignment operator.
   * @details Destroys the current plan and creates a new one based on the
   * other's configuration, using wisdom if available.
   * @param other The Plan object to copy from.
   * @return A reference to this object.
   */
  auto& operator=(const Plan& other) {
    _in = other._in;
    _out = other._out;
    _flag = other._flag;
    _direction = other._direction;
    _kinds = other._kinds;
    auto flag = _flag == Estimate ? Estimate : WisdomOnly;
    MakePlan(flag);
    return *this;
  }

  /**
   * @brief Move assignment operator.
   * @details Destroys the current plan and takes ownership of the other plan's
   * configuration. The moved-from plan is destroyed.
   * @param other The Plan object to move from.
   * @return A reference to this object.
   */
  auto& operator=(Plan&& other) {
    other.Destroy();
    _in = std::move(other._in);
    _out = std::move(other._out);
    _flag = std::move(other._flag);
    _direction = std::move(other._direction);
    _kinds = std::move(other._kinds);
    auto flag = _flag == Estimate ? Estimate : WisdomOnly;
    MakePlan(flag);
    return *this;
  }

  /**
   * @brief Destructor. Destroys the underlying FFTW plan.
   */
  ~Plan() { Destroy(); }

  /**
   * @brief Gets a const pointer to the underlying `fftw*_plan` handle.
   * @return The raw FFTW plan handle.
   */
  auto Pointer() const {
    if constexpr (NumericConcepts::Float<Real>) {
      return std::get<fftwf_plan>(_plan);
    }
    if constexpr (NumericConcepts::Double<Real>) {
      return std::get<fftw_plan>(_plan);
    }
    if constexpr (NumericConcepts::LongDouble<Real>) {
      return std::get<fftwl_plan>(_plan);
    }
  }

  /**
   * @brief Gets a non-const reference to the underlying `fftw*_plan` handle.
   * @return A reference to the raw FFTW plan handle.
   */
  auto& Pointer() {
    if constexpr (NumericConcepts::Float<Real>) {
      return std::get<fftwf_plan>(_plan);
    }
    if constexpr (NumericConcepts::Double<Real>) {
      return std::get<fftw_plan>(_plan);
    }
    if constexpr (NumericConcepts::LongDouble<Real>) {
      return std::get<fftwl_plan>(_plan);
    }
  }

  /**
   * @brief Checks if the underlying plan is null.
   * @return `true` if the plan has not been created or has been destroyed,
   * `false` otherwise.
   */
  auto IsNull() { return Pointer() == nullptr; }

  /**
   * @brief Calculates the normalization factor for an inverse transform.
   * @details The normalization factor is `1 / N`, where `N` is the total
   * logical size of the transform dimensions. For R2R transforms, this uses the
   * `LogicalDimension` of each transform kind.
   * @return The normalization factor, cast to the output value type.
   */
  auto Normalisation() const {
    int dim;
    if constexpr (NumericConcepts::Complex<InType> ||
                  NumericConcepts::Complex<OutType>) {
      dim = std::ranges::fold_left_first(_out.N(), std::multiplies<>()).value();
    } else {
      dim = std::ranges::fold_left_first(
                std::ranges::views::zip_transform(
                    [](auto n, auto kind) { return kind.LogicalDimension(n); },
                    _out.N(), Kinds()),
                std::multiplies<>())
                .value();
    }
    return static_cast<OutType>(1) / static_cast<OutType>(dim);
  }

  /**
   * @brief Executes the plan on the views provided during construction.
   */
  void Execute() { FFTWpp::Execute(Pointer()); }

  /**
   * @brief Executes the plan using new input and output data buffers.
   * @details This allows a plan to be reused with different data arrays,
   * provided they have the same layout and alignment characteristics as the
   * originals.
   * @tparam NewInView A range type for the new input data.
   * @tparam NewOutView A range type for the new output data.
   * @param in The new input data range.
   * @param out The new output data range.
   * @requires The value types of the new ranges must match the original ranges.
   */
  template <NumericConcepts::RealOrComplexWritableRange NewInView,
            NumericConcepts::RealOrComplexWritableRange NewOutView>
  requires NumericConcepts::SameRangeValueType<InView, NewInView> &&
           NumericConcepts::SameRangeValueType<OutView, NewOutView>
  void Execute(NewInView in, NewOutView out) {
    FFTWpp::Execute(Pointer(), in.data(), out.data());
  }

 private:
  /// @brief The input data view.
  View<InView> _in;
  /// @brief The output data view.
  View<OutView> _out;
  /// @brief The planner flag used for creation.
  Flag _flag;
  /// @brief The transform direction (for C2C transforms).
  std::variant<std::monostate, Direction> _direction;
  /// @brief The kinds of transform (for R2R transforms).
  std::variant<std::monostate, std::vector<RealKind>> _kinds;
  /// @brief A variant holding the precision-specific FFTW plan handle.
  std::variant<fftwf_plan, fftw_plan, fftwl_plan> _plan;

  /**
   * @brief Validates that input/output view dimensions are compatible for the
   * transform.
   * @return `true` if dimensions are valid, `false` otherwise.
   */
  auto CheckInputs() const {
    if (_in.Rank() != _out.Rank()) return false;
    if (_in.HowMany() != _out.HowMany()) return false;
    if constexpr (std::same_as<InType, OutType>) {
      return std::ranges::equal(_in.N(), _out.N());
    } else if constexpr (NumericConcepts::Complex<InType> &&
                         NumericConcepts::Real<OutType>) {
      // C2R: Output is real, last dimension of input is N/2 + 1
      return std::ranges::equal(
                 _in.N() | std::views::reverse | std::views::take(1),
                 _out.N() | std::views::reverse | std::views::take(1),
                 [](auto x, auto y) { return x == y / 2 + 1; }) &&
             std::ranges::equal(
                 _in.N() | std::views::reverse | std::views::drop(1),
                 _out.N() | std::views::reverse | std::views::drop(1));
    } else if constexpr (NumericConcepts::Real<InType> &&
                         NumericConcepts::Complex<OutType>) {
      // R2C: Input is real, last dimension of output is N/2 + 1
      return std::ranges::equal(
                 _in.N() | std::views::reverse | std::views::take(1),
                 _out.N() | std::views::reverse | std::views::take(1),
                 [](auto x, auto y) { return x / 2 + 1 == y; }) &&
             std::ranges::equal(
                 _in.N() | std::views::reverse | std::views::drop(1),
                 _out.N() | std::views::reverse | std::views::drop(1));
    }
  }

  /**
   * @brief Creates the underlying FFTW plan using the current configuration.
   * @details This function dispatches to the correct `FFTWpp::Plan` function
   * from `Core.h` based on the data types of the input and output views.
   * @param flag The planner flag to use for creation.
   */
  void MakePlan(Flag flag) {
    if constexpr (NumericConcepts::Complex<InType> &&
                  NumericConcepts::Complex<OutType>) {
      _plan = FFTWpp::Plan(_in.Rank(), _in.NPointer(), _in.HowMany(),
                           _in.DataPointer(), _in.EmbedPointer(), _in.Stride(),
                           _in.Dist(), _out.DataPointer(), _out.EmbedPointer(),
                           _out.Stride(), _out.Dist(),
                           std::get<Direction>(_direction), flag);
    } else if constexpr ((NumericConcepts::Complex<InType> &&
                          NumericConcepts::Real<OutType>)) {
      _plan = FFTWpp::Plan(_out.Rank(), _out.NPointer(), _out.HowMany(),
                           _in.DataPointer(), _in.EmbedPointer(), _in.Stride(),
                           _in.Dist(), _out.DataPointer(), _out.EmbedPointer(),
                           _out.Stride(), _out.Dist(), flag);
    } else if constexpr ((NumericConcepts::Real<InType> &&
                          NumericConcepts::Complex<OutType>)) {
      _plan = FFTWpp::Plan(_in.Rank(), _in.NPointer(), _in.HowMany(),
                           _in.DataPointer(), _in.EmbedPointer(), _in.Stride(),
                           _in.Dist(), _out.DataPointer(), _out.EmbedPointer(),
                           _out.Stride(), _out.Dist(), flag);
    } else if constexpr (NumericConcepts::Real<InType> &&
                         NumericConcepts::Real<OutType>) {
      auto kinds = std::vector<fftw_r2r_kind>();
      std::transform(
          Kinds().begin(), Kinds().end(), std::back_inserter(kinds),
          [](auto kind) { return static_cast<fftw_r2r_kind>(kind); });
      _plan = FFTWpp::Plan(_in.Rank(), _in.NPointer(), _in.HowMany(),
                           _in.DataPointer(), _in.EmbedPointer(), _in.Stride(),
                           _in.Dist(), _out.DataPointer(), _out.EmbedPointer(),
                           _out.Stride(), _out.Dist(), kinds.data(), flag);
    }
    assert(!IsNull());
  }

  /**
   * @brief Gets a view of the `RealKind`s for an R2R transform.
   * @return A view over the vector of `RealKind`s.
   * @requires The transform must be real-to-real.
   */
  auto Kinds() const
  requires(NumericConcepts::Real<InType> && NumericConcepts::Real<OutType>)
  {
    return std::ranges::views::all(std::get<std::vector<RealKind>>(_kinds));
  }

  /**
   * @brief Destroys the stored plan and resets the pointer to null.
   */
  void Destroy() {
    if (IsNull()) return;
    FFTWpp::Destroy(Pointer());
    Pointer() = nullptr;
  }
};

}  // namespace Ranges

}  // namespace FFTWpp

#endif  // FFTWPP_PLAN_GUARD_H