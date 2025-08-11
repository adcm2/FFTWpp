/**
 * @file Options.h
 * @brief Provides type-safe C++ wrappers for FFTW options and flags.
 *
 * This file defines classes and constants to represent various options for FFTW
 * transforms in a type-safe manner. This includes transform directions,
 * planner flags, and the kinds of real-to-real transforms, making the API
 * more robust and expressive at compile time.
 */
#ifndef FFTWPP_FLAGS_GUARD_H
#define FFTWPP_FLAGS_GUARD_H

#include "fftw3.h"

namespace FFTWpp {

/**
 * @class Direction
 * @brief A type-safe wrapper for FFTW transform directions.
 * @details Encapsulates `FFTW_FORWARD` and `FFTW_BACKWARD` to prevent misuse of
 * integer values in a type-safe, `constexpr` context.
 */
class Direction {
 public:
  /** @brief Default constructor. */
  constexpr Direction() = default;

  /**
   * @brief Constructs a Direction from an integer value.
   * @param direction The integer direction, must be `FFTW_FORWARD` or
   * `FFTW_BACKWARD`.
   */
  constexpr Direction(int direction) : _direction{direction} {
    assert(direction == FFTW_FORWARD || direction == FFTW_BACKWARD);
  }

  /**
   * @brief Implicit conversion to the underlying integer representation.
   * @return The integer value of the direction.
   */
  constexpr operator int() const { return _direction; }

  /** @brief Defaulted equality operator. */
  bool operator==(const Direction&) const = default;

 private:
  int _direction;
};

/** @brief Represents a forward transform (`-1`). */
constexpr auto Forward = Direction{FFTW_FORWARD};

/** @brief Represents a backward (inverse) transform (`+1`). */
constexpr auto Backward = Direction{FFTW_BACKWARD};

/**
 * @class Flag
 * @brief A type-safe wrapper for FFTW planner flags.
 * @details Encapsulates unsigned integer flags like `FFTW_MEASURE` and
 * provides a type-safe bitwise OR operator for combining them.
 */
class Flag {
 public:
  /** @brief Default constructor. */
  constexpr Flag() = default;

  /**
   * @brief Constructs a Flag from an unsigned integer value.
   * @param flag The unsigned integer flag.
   */
  constexpr Flag(unsigned flag) : _flag{flag} {}

  /**
   * @brief Implicit conversion to the underlying unsigned integer
   * representation.
   * @return The unsigned integer value of the flag.
   */
  constexpr operator unsigned() const { return _flag; }

  /** @brief Defaulted equality operator. */
  bool operator==(const Flag&) const = default;

 private:
  unsigned _flag;
};

/**
 * @brief Combines two planner flags using a bitwise OR operation.
 * @param lhs The left-hand side flag.
 * @param rhs The right-hand side flag.
 * @return A new Flag object representing the combined flags.
 */
constexpr auto operator|(Flag&& lhs, Flag&& rhs) {
  return Flag{static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs)};
}

/** @brief Planner flag: Specifies a simple estimate; no measurements are
 * performed. Quickest. */
constexpr auto Estimate = Flag{FFTW_ESTIMATE};

/** @brief Planner flag: Measures run-times of different algorithms to find the
 * best plan. */
constexpr auto Measure = Flag{FFTW_MEASURE};

/** @brief Planner flag: Like `Measure`, but tries a wider range of algorithms.
 * Slower. */
constexpr auto Patient = Flag{FFTW_PATIENT};

/** @brief Planner flag: Like `Patient`, but even more exhaustive. Very slow. */
constexpr auto Exhaustive = Flag{FFTW_EXHAUSTIVE};

/** @brief Planner flag: Only uses wisdom; does not run any measurements. Fails
 * if no wisdom is available. */
constexpr auto WisdomOnly = Flag{FFTW_WISDOM_ONLY};

/** @brief Planner flag: Allows the input array to be overwritten during
 * planning. */
constexpr auto DestroyInput = Flag{FFTW_DESTROY_INPUT};

/** @brief Planner flag: The input array is not overwritten during planning
 * (default). */
constexpr auto PreserveInput = Flag{FFTW_PRESERVE_INPUT};

/** @brief Planner flag: Indicates that input/output arrays may not be suitably
 * aligned. */
constexpr auto Unaligned = Flag{FFTW_UNALIGNED};

/**
 * @class RealKind
 * @brief A type-safe wrapper for the kind of real-to-real (`r2r`) transform.
 * @details Encapsulates the `fftw_r2r_kind` enum and provides helper methods
 * for finding inverse transforms and their logical dimensions for
 * normalization.
 */
class RealKind {
 public:
  /** @brief Default constructor. */
  constexpr RealKind() = default;

  /**
   * @brief Constructs a RealKind from an `fftw_r2r_kind` enum value.
   * @param kind The kind of real-to-real transform.
   */
  constexpr RealKind(fftw_r2r_kind kind) : _kind{kind} {}

  /**
   * @brief Implicit conversion to the underlying `fftw_r2r_kind`
   * representation.
   * @return The `fftw_r2r_kind` enum value.
   */
  constexpr operator fftw_r2r_kind() const { return _kind; }

  /** @brief Defaulted equality operator. */
  bool operator==(const RealKind&) const = default;

  /**
   * @brief Gets the inverse of the current transform kind.
   * @details For example, the inverse of a DCT-III (`REDFT10`) is a DCT-II
   * (`REDFT01`). Some transforms like DHT and REDFT00 are their own inverse.
   * @return The `RealKind` of the inverse transform.
   */
  constexpr auto Inverse() const {
    switch (_kind) {
      case FFTW_R2HC:
        return RealKind{FFTW_HC2R};
      case FFTW_HC2R:
        return RealKind{FFTW_R2HC};
      case FFTW_DHT:
        return RealKind{FFTW_DHT};
      case FFTW_REDFT00:
        return RealKind{FFTW_REDFT00};
      case FFTW_REDFT10:
        return RealKind{FFTW_REDFT01};
      case FFTW_REDFT01:
        return RealKind{FFTW_REDFT10};
      case FFTW_REDFT11:
        return RealKind{FFTW_REDFT11};
      case FFTW_RODFT00:
        return RealKind{FFTW_RODFT00};
      case FFTW_RODFT10:
        return RealKind{FFTW_RODFT01};
      case FFTW_RODFT01:
        return RealKind{FFTW_RODFT10};
      case FFTW_RODFT11:
        return RealKind{FFTW_RODFT11};
      default:
        return RealKind{FFTW_HC2R};
    }
  }

  /**
   * @brief Gets the logical size of the transform for normalization purposes.
   * @details For a given dimension of size `n`, the logical size depends on the
   * transform type. For example, for a DCT-I (`REDFT00`), the logical size is
   * `2*(n-1)`. For halfcomplex transforms, it is `n`.
   * @param n The physical size of the transform dimension.
   * @return The logical size of the dimension.
   */
  constexpr auto LogicalDimension(int n) const {
    switch (_kind) {
      case FFTW_R2HC:
        return n;
      case FFTW_HC2R:
        return n;
      case FFTW_DHT:
        return n;
      case FFTW_REDFT00:
        return 2 * (n - 1);
      case FFTW_REDFT10:
        return 2 * n;
      case FFTW_REDFT01:
        return 2 * n;
      case FFTW_REDFT11:
        return 2 * n;
      case FFTW_RODFT00:
        return 2 * (n + 1);
      case FFTW_RODFT10:
        return 2 * n;
      case FFTW_RODFT01:
        return 2 * n;
      case FFTW_RODFT11:
        return 2 * n;
      default:
        return n;
    }
  }

 private:
  fftw_r2r_kind _kind;
};

/** @brief Real-to-halfcomplex transform. */
constexpr auto R2HC = RealKind{FFTW_R2HC};

/** @brief Halfcomplex-to-real transform. */
constexpr auto HC2R = RealKind{FFTW_HC2R};

/** @brief Discrete Hartley Transform. */
constexpr auto DHT = RealKind{FFTW_DHT};

/** @brief Discrete Cosine Transform, type I (DCT-I). */
constexpr auto REDFT00 = RealKind{FFTW_REDFT00};

/** @brief Discrete Cosine Transform, type II (DCT-II), often called "the" DCT.
 */
constexpr auto REDFT10 = RealKind{FFTW_REDFT10};

/** @brief Discrete Cosine Transform, type III (DCT-III), often called "the"
 * IDCT. */
constexpr auto REDFT01 = RealKind{FFTW_REDFT01};

/** @brief Discrete Cosine Transform, type IV (DCT-IV). */
constexpr auto REDFT11 = RealKind{FFTW_REDFT11};

/** @brief Discrete Sine Transform, type I (DST-I). */
constexpr auto RODFT00 = RealKind{FFTW_RODFT00};

/** @brief Discrete Sine Transform, type II (DST-II). */
constexpr auto RODFT10 = RealKind{FFTW_RODFT10};

/** @brief Discrete Sine Transform, type III (DST-III). */
constexpr auto RODFT01 = RealKind{FFTW_RODFT01};

/** @brief Discrete Sine Transform, type IV (DST-IV). */
constexpr auto RODFT11 = RealKind{FFTW_RODFT11};

}  // namespace FFTWpp

#endif  //  FFTWPP_FLAGS_GUARD_H