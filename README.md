# FFTWpp: A Modern C++ Wrapper for FFTW3

## Introduction

**FFTWpp** is a header-only C++ library that provides a modern, type-safe, and user-friendly wrapper around the acclaimed [FFTW3](http://www.fftw.org/) library. It leverages C++20 features like concepts and ranges to create an expressive and safe API, abstracting away much of the boilerplate and potential pitfalls of the underlying C interface.

### Key Features 

* **Modern C++20 Idioms**: Utilizes concepts for compile-time validation and ranges/views for expressive data handling.
* **RAII for Plan Management**: The `FFTWpp::Ranges::Plan` class automatically manages the lifecycle of `fftw_plan` objects, ensuring resources are created and destroyed correctly.
* **Type Safety**: Replaces raw integer flags and enums with strongly-typed classes like `Flag`, `Direction`, and `RealKind`, preventing common errors.
* **Automatic Precision Selection**: Uses templates to automatically select the correct FFTW precision (`float`, `double`, `long double`) based on your data types, so you don't have to call `fftwf_`, `fftw_`, or `fftwl_` functions manually.
* **Simplified API**: Greatly simplifies creating both simple and advanced transform plans.

---

## Core Concepts

The library is built around a few central ideas that work together to provide a seamless experience.

### Layout and View

The foundation of data handling in FFTWpp is the distinction between a data's shape and the data itself.

* `FFTWpp::Ranges::Layout`: An object that describes the *shape* and memory organization of your data. It holds all the parameters needed by FFTW's advanced interface, such as rank (dimensionality), dimensions, strides, and the number of transforms to perform.
* `FFTWpp::Ranges::View`: This class combines a `Layout` with an actual range of data (like an `FFTWpp::vector` or `std::span`). It's the primary object you'll pass to the `Plan` constructor, as it contains both the data and the description of its shape.

### Plan

The `FFTWpp::Ranges::Plan` is the central workhorse of the library. It's an RAII object that encapsulates an `fftw_plan`.

1.  You **construct** it with input and output `View`s.
2.  The constructor **creates** the underlying `fftw_plan`, choosing the optimal algorithm (and generating "wisdom" if requested).
3.  You call `Execute()` on the `Plan` object.
4.  When the `Plan` object goes out of scope, its **destructor** automatically calls `fftw_destroy_plan`.

### Options

To make configuration readable and safe, FFTWpp provides wrapper classes for common options:

* `FFTWpp::Direction`: For `Forward` and `Backward` transforms.
* `FFTWpp::Flag`: For planner flags like `Measure`, `Estimate`, and `Patient`. These can be combined with the `|` operator.
* `FFTWpp::RealKind`: For specifying the type of real-to-real transforms (e.g., `REDFT00` for a DCT-I).

---

## Quick Start 

Here is a complete example of performing a 1D complex-to-complex DFT and its inverse, and verifying the result.

```cpp
#include <iostream>
#include "FFTWpp/Plan.h"
#include "FFTWpp/Utility.h"

int main() {
    // 1. Define transform parameters
    const int N = 16;
    using Real = double;
    using Complex = std::complex<Real>;

    // 2. Create data containers using the FFTW-aligned allocator
    auto in = FFTWpp::vector<Complex>(N);
    auto temp = FFTWpp::vector<Complex>(N);
    auto out = FFTWpp::vector<Complex>(N);

    // 3. Populate input with random data and make a copy for later verification
    FFTWpp::RandomiseValues(in);
    std::copy(in.begin(), in.end(), out.begin());

    std::cout << "Successfully set up data containers." << std::endl;

    // 4. Create Views for the data.
    // A View combines the data range with its layout/shape.
    auto inView = FFTWpp::Ranges::View(in);
    auto tempView = FFTWpp::Ranges::View(temp);
    auto outView = FFTWpp::Ranges::View(out);

    // 5. Create and execute the forward plan (in -> temp)
    {
        auto planForward = FFTWpp::Ranges::Plan(inView, tempView, FFTWpp::Measure, FFTWpp::Forward);
        std::cout << "Executing forward DFT..." << std::endl;
        planForward.Execute();
    }

    {
        // 6. Create and execute the backward plan (temp -> out)
        FFTWpp::Ranges::Plan planBackward(tempView, outView, FFTWpp::Measure, FFTWpp::Backward);
        std::cout << "Executing backward DFT..." << std::endl;
        planBackward.Execute();

        // 7. Normalize the result of the backward transform
        auto norm = planBackward.Normalisation();
        std::transform(out.begin(), out.end(), out.begin(),
                       [norm](auto x){ return x * norm; });

        // 8. Verify that the result matches the original input
        if (FFTWpp::CheckValues(in, out, 1.0)) {
            std::cout << "Success! The inverse transform matches the original data." << std::endl;
        } else {
            std::cout << "Failure! The inverse transform does not match." << std::endl;
        }
    }

    // 9. Clean up global FFTW data only after every plan has been destroyed.
    FFTWpp::CleanUp();
    return 0;
}
```

## Advanced Usage

### Real Transforms (R2C, C2R, R2R)

The library transparently handles real-to-complex and complex-to-real transforms. The correct transform type is inferred from the data types of the input and output views.

```cpp

// For an R2C transform
FFTWpp::vector<double> r_in(N);
FFTWpp::vector<std::complex<double>> c_out(N / 2 + 1);
auto r_in_view = FFTWpp::Ranges::View(r_in);
auto c_out_view = FFTWpp::Ranges::View(c_out);

// The plan constructor automatically selects an R2C transform
auto planR2C = FFTWpp::Ranges::Plan(r_in_view, c_out_view, FFTWpp::Measure);

```

For real-to-real (R2R) transforms, you provide the kind(s) of transform as extra arguments.

```cpp
// For an R2R transform (e.g., a DCT-II)
FFTWpp::vector<double> r_in(N);
FFTWpp::vector<double> r_out(N);
auto r_in_view = FFTWpp::Ranges::View(r_in);
auto r_out_view = FFTWpp::Ranges::View(r_out);

// Provide the kind(s) as the last arguments
auto planR2R = FFTWpp::Ranges::Plan(r_in_view, r_out_view, FFTWpp::Measure, FFTWpp::REDFT10);

```

## Wisdom Management

Wisdom allows FFTW to reuse information about optimal plans, speeding up initialization.

```cpp

// Generate wisdom for a given transform size without executing it
FFTWpp::Ranges::Layout layout{128, 128};
FFTWpp::GenerateWisdom<Complex, Complex>(layout, layout, FFTWpp::Patient);

// Save all accumulated wisdom to a file
FFTWpp::ExportWisdom("my_fftw_wisdom.dat");

// Load wisdom at the start of your application
FFTWpp::ImportWisdom("my_fftw_wisdom.dat");

```

`ExportWisdom` and `ImportWisdom` operate on FFTW's double-precision wisdom
store. `ForgetWisdom` clears the float, double, and long-double stores.
`CleanUp` releases serial FFTW process-global planner state for all three
precisions; it does not destroy plans or clean up the optional FFTW threads
interfaces. Destroy every live plan before calling it.

## Testing

See [docs/testing.md](docs/testing.md) for the test inventory and sanitizer
workflow.

## Library Structure

`Core.h`: Contains the low-level, precision-aware C++ wrappers for the core ```fftw3.h``` functions. This is the foundation upon which the higher-level abstractions are built.

`Options.h`: Defines the type-safe classes ```Direction```, ```Flag```, and ```RealKind``` for configuring transforms.

`Views.h`: Defines the ```Layout``` and ```View``` classes for describing the shape and location of your data.

`Plan.h`: Contains the primary user-facing ```FFTWpp::Ranges::Plan``` class which provides RAII-based plan management.

`Wisdom.h`: Provides utilities for importing, exporting, and generating FFTW wisdom.

`Utility.h`: Includes helper functions for testing and setup, such as ```RandomiseValues``` and ```CheckValues```.
