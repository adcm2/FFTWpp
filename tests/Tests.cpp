#include <gtest/gtest.h>

#include <filesystem>

#include "Test1D.h"

namespace {

template <typename Real>
class PlanOwnership : public testing::Test {};

using Precisions = testing::Types<float, double, long double>;
TYPED_TEST_SUITE(PlanOwnership, Precisions);

TYPED_TEST(PlanOwnership, CopyConstructionAndAssignmentOwnDistinctPlans) {
  using Complex = std::complex<TypeParam>;
  auto in = FFTWpp::vector<Complex>(8);
  auto out = FFTWpp::vector<Complex>(8);
  auto inView = FFTWpp::Ranges::View(in);
  auto outView = FFTWpp::Ranges::View(out);
  auto source =
      FFTWpp::Ranges::Plan(inView, outView, FFTWpp::Estimate, FFTWpp::Forward);
  auto destination =
      FFTWpp::Ranges::Plan(inView, outView, FFTWpp::Estimate, FFTWpp::Backward);
  auto oldDestination = destination.Pointer();

  auto copy(source);
  EXPECT_NE(copy.Pointer(), nullptr);
  EXPECT_NE(copy.Pointer(), source.Pointer());

  destination = source;
  EXPECT_NE(destination.Pointer(), nullptr);
  EXPECT_NE(destination.Pointer(), source.Pointer());
  EXPECT_NE(destination.Pointer(), oldDestination);
}

TYPED_TEST(PlanOwnership, MoveConstructionAndAssignmentTransferOwnership) {
  using Complex = std::complex<TypeParam>;
  auto in = FFTWpp::vector<Complex>(8);
  auto out = FFTWpp::vector<Complex>(8);
  auto inView = FFTWpp::Ranges::View(in);
  auto outView = FFTWpp::Ranges::View(out);
  auto source =
      FFTWpp::Ranges::Plan(inView, outView, FFTWpp::Estimate, FFTWpp::Forward);
  auto sourcePointer = source.Pointer();
  auto moved(std::move(source));

  EXPECT_EQ(moved.Pointer(), sourcePointer);
  EXPECT_TRUE(source.IsNull());

  auto destination =
      FFTWpp::Ranges::Plan(inView, outView, FFTWpp::Estimate, FFTWpp::Backward);
  auto movedPointer = moved.Pointer();
  destination = std::move(moved);
  EXPECT_EQ(destination.Pointer(), movedPointer);
  EXPECT_TRUE(moved.IsNull());
}

TYPED_TEST(PlanOwnership, SelfAndRepeatedAssignmentRemainValid) {
  using Complex = std::complex<TypeParam>;
  auto in = FFTWpp::vector<Complex>(8);
  auto out = FFTWpp::vector<Complex>(8);
  auto inView = FFTWpp::Ranges::View(in);
  auto outView = FFTWpp::Ranges::View(out);
  auto source =
      FFTWpp::Ranges::Plan(inView, outView, FFTWpp::Estimate, FFTWpp::Forward);
  auto destination =
      FFTWpp::Ranges::Plan(inView, outView, FFTWpp::Estimate, FFTWpp::Backward);

  auto pointer = destination.Pointer();
  destination = destination;
  EXPECT_EQ(destination.Pointer(), pointer);
  destination = std::move(destination);
  EXPECT_EQ(destination.Pointer(), pointer);

  for (int i = 0; i < 32; ++i) destination = source;
  for (int i = 0; i < 32; ++i) {
    auto temporary = FFTWpp::Ranges::Plan(inView, outView, FFTWpp::Estimate,
                                          FFTWpp::Backward);
    auto temporaryPointer = temporary.Pointer();
    destination = std::move(temporary);
    EXPECT_EQ(destination.Pointer(), temporaryPointer);
    EXPECT_TRUE(temporary.IsNull());
  }
}

TYPED_TEST(PlanOwnership, FailedCopyReplacementPreservesDestination) {
  using Complex = std::complex<TypeParam>;
  auto layout = FFTWpp::Ranges::Layout(8);
  FFTWpp::ForgetWisdom();
  FFTWpp::GenerateWisdom<Complex, Complex>(layout, layout, FFTWpp::Measure);

  auto in = FFTWpp::vector<Complex>(8);
  auto out = FFTWpp::vector<Complex>(8);
  auto inView = FFTWpp::Ranges::View(in);
  auto outView = FFTWpp::Ranges::View(out);
  auto source = FFTWpp::Ranges::Plan(inView, outView, FFTWpp::WisdomOnly,
                                     FFTWpp::Forward);
  auto destination =
      FFTWpp::Ranges::Plan(inView, outView, FFTWpp::Estimate, FFTWpp::Backward);
  auto destinationPointer = destination.Pointer();
  FFTWpp::ForgetWisdom();

  EXPECT_THROW(destination = source, std::runtime_error);
  EXPECT_EQ(destination.Pointer(), destinationPointer);
}

template <typename Real>
class WisdomGeneration : public testing::Test {};

TYPED_TEST_SUITE(WisdomGeneration, Precisions);

TYPED_TEST(WisdomGeneration, ComplexForwardAndBackwardPlansAreReusable) {
  using Complex = std::complex<TypeParam>;
  auto layout = FFTWpp::Ranges::Layout(8);
  FFTWpp::ForgetWisdom();
  FFTWpp::GenerateWisdom<Complex, Complex>(layout, layout, FFTWpp::Measure);

  auto in = FFTWpp::vector<Complex>(8);
  auto out = FFTWpp::vector<Complex>(8);
  auto inView = FFTWpp::Ranges::View(in);
  auto outView = FFTWpp::Ranges::View(out);
  EXPECT_NO_THROW(FFTWpp::Ranges::Plan(inView, outView, FFTWpp::WisdomOnly,
                                      FFTWpp::Forward));
  EXPECT_NO_THROW(FFTWpp::Ranges::Plan(outView, inView, FFTWpp::WisdomOnly,
                                      FFTWpp::Backward));
}

TYPED_TEST(WisdomGeneration, RealComplexPlansAreReusableInBothDirections) {
  using Complex = std::complex<TypeParam>;
  auto realLayout = FFTWpp::Ranges::Layout(8);
  auto complexLayout = FFTWpp::Ranges::Layout(5);
  FFTWpp::ForgetWisdom();
  FFTWpp::GenerateWisdom<TypeParam, Complex>(realLayout, complexLayout,
                                             FFTWpp::Measure);

  auto real = FFTWpp::vector<TypeParam>(8);
  auto complex = FFTWpp::vector<Complex>(5);
  auto realView = FFTWpp::Ranges::View(real);
  auto complexView = FFTWpp::Ranges::View(complex);
  EXPECT_NO_THROW(
      FFTWpp::Ranges::Plan(realView, complexView, FFTWpp::WisdomOnly));
  EXPECT_NO_THROW(
      FFTWpp::Ranges::Plan(complexView, realView, FFTWpp::WisdomOnly));
}

TYPED_TEST(WisdomGeneration, RealBackwardPlanUsesInverseKinds) {
  auto layout = FFTWpp::Ranges::Layout(8);
  FFTWpp::ForgetWisdom();
  FFTWpp::GenerateWisdom<TypeParam, TypeParam>(
      layout, layout, {FFTWpp::REDFT10}, FFTWpp::Measure);

  auto in = FFTWpp::vector<TypeParam>(8);
  auto out = FFTWpp::vector<TypeParam>(8);
  EXPECT_NO_THROW(FFTWpp::Ranges::Plan(
      FFTWpp::Ranges::View(in), FFTWpp::Ranges::View(out),
      FFTWpp::WisdomOnly, FFTWpp::REDFT01));
}

TEST(WisdomIO, DoublePrecisionExportAndImportReportSuccessCorrectly) {
  const auto path =
      std::filesystem::temp_directory_path() / "fftwpp-test-wisdom.dat";
  FFTWpp::ForgetWisdom();
  auto layout = FFTWpp::Ranges::Layout(8);
  FFTWpp::GenerateWisdom<std::complex<double>, std::complex<double>>(
      layout, layout, FFTWpp::Measure);

  EXPECT_NO_THROW(FFTWpp::ExportWisdom(path.string()));
  FFTWpp::ForgetWisdom();
  EXPECT_NO_THROW(FFTWpp::ImportWisdom(path.string()));
  EXPECT_TRUE(std::filesystem::remove(path));
}

}  // namespace

// 1D C2C tests
TEST(Test1DC2C, FLOAT) {
  using Complex = std::complex<float>;
  auto result = Test1D<Complex, Complex>();
  EXPECT_TRUE(result);
}

TEST(Test1DC2C, DOUBLE) {
  using Complex = std::complex<double>;
  auto result = Test1D<Complex, Complex>();
  EXPECT_TRUE(result);
}

TEST(Test1DC2C, LONGDOUBLE) {
  using Complex = std::complex<long double>;
  auto result = Test1D<Complex, Complex>();
  EXPECT_TRUE(result);
}

// 1D R2C tests
TEST(Test1DR2C, FLOAT) {
  using Real = float;
  using Complex = std::complex<Real>;
  auto result = Test1D<Real, Complex>();
  EXPECT_TRUE(result);
}

TEST(Test1DR2C, DOUBLE) {
  using Real = double;
  using Complex = std::complex<Real>;
  auto result = Test1D<Real, Complex>();
  EXPECT_TRUE(result);
}

TEST(Test1DR2C, LONGDOUBLE) {
  using Real = long double;
  using Complex = std::complex<Real>;
  auto result = Test1D<Real, Complex>();
  EXPECT_TRUE(result);
}

// 1D R2R tests
TEST(Test1DR2R, FLOAT) {
  using Real = float;
  auto result = Test1D<Real, Real>();
  EXPECT_TRUE(result);
}

TEST(Test1DR2R, DOUBLE) {
  using Real = double;
  auto result = Test1D<Real, Real>();
  EXPECT_TRUE(result);
}

TEST(Test1DR2R, LONGDOUBLE) {
  using Real = long double;
  auto result = Test1D<Real, Real>();
  EXPECT_TRUE(result);
}
