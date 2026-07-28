# Testing

The `Tests` executable uses GoogleTest. CTest discovers each typed precision
case separately. Its custom `main` calls `FFTWpp::CleanUp()` only after all
tests, and therefore all test-local plans, have finished.

## Plan ownership tests

- `PlanOwnership.CopyConstructionAndAssignmentOwnDistinctPlans` checks that
  copying creates an independent FFTW plan and copy assignment replaces an
  already-owned destination plan.
- `PlanOwnership.MoveConstructionAndAssignmentTransferOwnership` checks that
  move construction and move assignment transfer the exact FFTW handle, leave
  the source null, and replace an already-owned destination plan.
- `PlanOwnership.SelfAndRepeatedAssignmentRemainValid` checks self-copy,
  self-move, repeated copy assignment, and repeated move assignment. Leak and
  double-destruction regressions are detected by the sanitizer workflow.
- `PlanOwnership.FailedCopyReplacementPreservesDestination` removes wisdom
  needed to copy a `WisdomOnly` plan and checks that the failed replacement
  leaves the destination's existing handle intact.

Each ownership test runs for `float`, `double`, and `long double`.

## Wisdom tests

- `WisdomGeneration.ComplexForwardAndBackwardPlansAreReusable` generates
  complex-to-complex wisdom and recreates both directions with
  `WisdomOnly`.
- `WisdomGeneration.RealComplexPlansAreReusableInBothDirections` generates
  real-to-complex and complex-to-real wisdom and recreates both plans with
  `WisdomOnly`.
- `WisdomGeneration.RealBackwardPlanUsesInverseKinds` verifies that
  real-to-real wisdom contains the inverse kind for the backward plan.
- `WisdomIO.DoublePrecisionExportAndImportReportSuccessCorrectly` checks the
  documented double-precision file API and its success return handling.

The wisdom generation tests run for `float`, `double`, and `long double`.

## Transform correctness tests

- `Test1DC2C` checks a measured complex forward/backward round trip.
- `Test1DR2C` checks a measured real/complex forward/backward round trip.
- `Test1DR2R` checks a measured real-to-real forward/inverse round trip.

Each transform correctness test runs for `float`, `double`, and `long double`.

`OdrInclude.cpp` includes the public aggregate header in a second translation
unit so the test link detects non-inline, header-defined API functions.
