// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// gfx's side of error handling: the core's Status, Result and VkResult bridge
// under gfx's names, and the VG_* macros. The core's own tests cover them
// themselves.

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/gfx/core/log.hpp"
#include "volumetric_kit/gfx/core/result.hpp"

namespace vg = volumetric_kit::gfx;
namespace vkc = volumetric_kit::core;

// A Status from gfx passes to recon or calib unchanged: it is the core's type,
// and gfx's VkResult bridge is the core's functions, not a copy of them.
TEST(Status, IsTheCoresType) {
  static_assert(std::is_same_v<vg::Status, vkc::Status>);
  static_assert(std::is_same_v<vg::Result<int>, vkc::Result<int>>);
  static_assert(&vg::vk_error == &vkc::vk_error);
  static_assert(&vg::vk_result == &vkc::vk_result);
  EXPECT_EQ(to_string(vg::Status::Code::Backend), "Backend");
  EXPECT_EQ(vg::to_string(vg::Status::Code::Backend), "Backend");
}

TEST(Status, VkErrorBuildsABackendStatus) {
  const vg::Status status =
      vg::vk_error(VK_ERROR_OUT_OF_DEVICE_MEMORY, "context");
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.domain(), vg::Status::Code::Backend);
  EXPECT_EQ(status.detail(), VK_ERROR_OUT_OF_DEVICE_MEMORY);
  EXPECT_EQ(status.message(), "context");
}

// vk_result reads the VkResult back, and nothing from another domain.
TEST(Status, VkResultReadsTheVkResultBack) {
  EXPECT_EQ(vg::vk_result(vg::vk_error(VK_ERROR_DEVICE_LOST, "lost")),
            VK_ERROR_DEVICE_LOST);
  EXPECT_EQ(vg::vk_result(vg::vk_error(VK_TIMEOUT, "wait")), VK_TIMEOUT);
  EXPECT_EQ(vg::vk_result(vg::Status{}), std::nullopt);
  EXPECT_EQ(vg::vk_result(vg::Status::invalid_argument("bad")), std::nullopt);
}

// A backend detail wider than 32 bits is no VkResult; converting it to the
// enum would be undefined (UBSan's enum check).
TEST(Status, VkResultRejectsADetailWiderThan32Bits) {
  EXPECT_EQ(vg::vk_result(vg::Status::backend_error(std::int64_t{1} << 40, "")),
            std::nullopt);
  EXPECT_EQ(vg::vk_result(vg::Status::backend_error(
                std::numeric_limits<std::int64_t>::min(), "")),
            std::nullopt);
  EXPECT_EQ(vg::vk_result(vg::Status::backend_error(
                std::numeric_limits<std::int32_t>::min(), "")),
            static_cast<VkResult>(std::numeric_limits<std::int32_t>::min()));
}

TEST(Status, ToStringNamesAreStable) {
  EXPECT_EQ(to_string(vg::Status::Code::Ok), "Ok");
  EXPECT_EQ(to_string(vg::Status::Code::InvalidArgument), "InvalidArgument");
  EXPECT_EQ(vg::to_string(VK_ERROR_DEVICE_LOST), "VK_ERROR_DEVICE_LOST");
  EXPECT_EQ(vg::to_string(VK_SUCCESS), "VK_SUCCESS");
  // A non-core code still in the table resolves to its real name.
  EXPECT_EQ(vg::to_string(VK_ERROR_FRAGMENTED_POOL),
            "VK_ERROR_FRAGMENTED_POOL");
  // An unlisted code falls back to a stable, never-empty label.
  EXPECT_FALSE(vg::to_string(static_cast<VkResult>(0x7fffffff)).empty());
}

namespace {

// A non-copyable, movable type to prove Result<T> works for move-only T
// (Result<Instance> / Result<Device> rely on this).
struct MoveOnly {
  int value;
  explicit MoveOnly(int v) : value(v) {}
  MoveOnly(MoveOnly&&) = default;
  MoveOnly& operator=(MoveOnly&&) = default;
  MoveOnly(const MoveOnly&) = delete;
  MoveOnly& operator=(const MoveOnly&) = delete;
};

vg::Status try_inner(bool fail) {
  if (fail) return vg::vk_error(VK_ERROR_UNKNOWN, "inner failed");
  return vg::Status{};
}

vg::Status try_outer(bool fail) {
  VG_TRY(try_inner(fail));
  return vg::Status{};
}

VkResult passthrough(VkResult r) { return r; }

vg::Status vk_try_caller(VkResult r) {
  VG_VK_TRY(passthrough(r));
  return vg::Status{};
}

}  // namespace

TEST(Try, PropagatesErrorAndPassesSuccess) {
  EXPECT_EQ(vg::vk_result(try_outer(true)), VK_ERROR_UNKNOWN);
  EXPECT_TRUE(try_outer(false).ok());
}

TEST(VkTry, PassesSuccessAndPropagatesFailureWithExprContext) {
  EXPECT_TRUE(vk_try_caller(VK_SUCCESS).ok());

  const vg::Status status = vk_try_caller(VK_ERROR_DEVICE_LOST);
  EXPECT_EQ(vg::vk_result(status), VK_ERROR_DEVICE_LOST);
  // VG_VK_TRY stringifies the expression (#expr) as the context message.
  EXPECT_NE(status.message().find("passthrough(r)"), std::string::npos);
}

// Every success code other than VK_SUCCESS is a failure to VG_VK_TRY.
TEST(VkTry, TreatsPositiveSuccessCodesAsFailures) {
  EXPECT_EQ(vg::vk_result(vk_try_caller(VK_SUBOPTIMAL_KHR)), VK_SUBOPTIMAL_KHR);
}

namespace {

vg::Result<int> make_int(bool fail) {
  if (fail) return vg::vk_error(VK_ERROR_UNKNOWN, "no int");
  return 21;
}

// Two VG_ASSIGNs in one scope prove the __COUNTER__-keyed temporaries don't
// collide; a failing one early-returns its Status.
vg::Result<int> sum_two(bool fail_second) {
  VG_ASSIGN(int a, make_int(false));
  VG_ASSIGN(int b, make_int(fail_second));
  return a + b;
}

// VG_ASSIGN must move the value out, so it works for move-only T.
vg::Status take_move_only() {
  VG_ASSIGN(MoveOnly m, vg::Result<MoveOnly>(MoveOnly{5}));
  if (m.value != 5) return vg::Status::invalid_argument("wrong");
  return vg::Status{};
}

}  // namespace

TEST(VgAssign, BindsValueOnSuccessAndPropagatesError) {
  const vg::Result<int> ok = sum_two(/*fail_second=*/false);
  ASSERT_TRUE(ok.ok());
  EXPECT_EQ(ok.value(), 42);

  const vg::Result<int> err = sum_two(/*fail_second=*/true);
  EXPECT_FALSE(err.ok());
  EXPECT_EQ(vg::vk_result(err.status()), VK_ERROR_UNKNOWN);
}

TEST(VgAssign, MovesMoveOnlyValue) { EXPECT_TRUE(take_move_only().ok()); }

// Reading the value of an error Result is a contract violation: it logs and
// aborts (SIGABRT). The "DeathTest" suffix makes gtest run this in isolation.
TEST(ResultDeathTest, ValueOnErrorAborts) {
  const vg::Result<int> result(vg::vk_error(VK_ERROR_DEVICE_LOST, "lost"));
  EXPECT_DEATH(
      {
        vg::set_log_handler({});  // ensure the abort message reaches stderr
        (void)result.value();
      },
      "contract check failed");
}

// VG_CHECK reports the condition as written, macros unexpanded.
TEST(CheckDeathTest, FailureNamesTheConditionAsWritten) {
  const unsigned n = VK_UUID_SIZE + 1;
  EXPECT_DEATH(
      {
        vg::set_log_handler({});
        VG_CHECK(n <= VK_UUID_SIZE, "too long");
      },
      "n <= VK_UUID_SIZE");
}
