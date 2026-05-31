// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <string>
#include <utility>

#include "volumetric_kit/gfx/core/log.hpp"
#include "volumetric_kit/gfx/core/result.hpp"

namespace vg = volumetric_kit::gfx;

TEST(Status, DefaultConstructedIsOk) {
  vg::Status status;
  EXPECT_TRUE(status.ok());
  EXPECT_TRUE(static_cast<bool>(status));
  EXPECT_EQ(status.domain(), vg::Status::Code::Ok);
  EXPECT_EQ(status.code(), VK_SUCCESS);
}

TEST(Status, ErrorCarriesDomainCodeAndMessage) {
  vg::Status status = vg::Status::error(VK_ERROR_OUT_OF_HOST_MEMORY, "boom");
  EXPECT_FALSE(status.ok());
  EXPECT_FALSE(static_cast<bool>(status));
  EXPECT_EQ(status.domain(), vg::Status::Code::Vulkan);
  EXPECT_EQ(status.code(), VK_ERROR_OUT_OF_HOST_MEMORY);
  EXPECT_EQ(status.message(), "boom");
}

TEST(Status, VkErrorHelperBuildsVulkanDomainError) {
  vg::Status status = vg::vk_error(VK_ERROR_OUT_OF_DEVICE_MEMORY, "context");
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.domain(), vg::Status::Code::Vulkan);
  EXPECT_EQ(status.code(), VK_ERROR_OUT_OF_DEVICE_MEMORY);
  EXPECT_EQ(status.message(), "context");
}

// Non-Vulkan domain factories: not ok, carry their domain, and report a
// VK_SUCCESS code() (there is no honest VkResult for them).
TEST(Status, DomainFactoriesCarryDomainAndOkCode) {
  using Code = vg::Status::Code;
  const std::pair<vg::Status, Code> cases[] = {
      {vg::Status::invalid_argument("a"), Code::InvalidArgument},
      {vg::Status::not_found("b"), Code::NotFound},
      {vg::Status::unsupported("c"), Code::Unsupported},
      {vg::Status::out_of_memory("d"), Code::OutOfMemory},
      {vg::Status::io_error("e"), Code::IoError},
  };
  for (const auto& [status, expected] : cases) {
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.domain(), expected);
    EXPECT_EQ(status.code(), VK_SUCCESS);
    EXPECT_FALSE(status.message().empty());
  }
}

// A Result built from a non-Vulkan domain error must NOT abort (the
// Result(Status) VG_CHECK keys on ok(), which is domain-driven) and must report
// the error faithfully.
TEST(Result, BuildsFromNonVulkanDomainErrorWithoutAbort) {
  vg::Result<int> result(vg::Status::invalid_argument("bad"));
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().domain(), vg::Status::Code::InvalidArgument);
  EXPECT_EQ(result.status().message(), "bad");
}

TEST(Status, ToStringNamesAreStable) {
  EXPECT_EQ(vg::to_string(vg::Status::Code::Ok), "Ok");
  EXPECT_EQ(vg::to_string(vg::Status::Code::InvalidArgument),
            "InvalidArgument");
  EXPECT_EQ(vg::to_string(vg::Status::Code::Vulkan), "Vulkan");
  EXPECT_EQ(vg::to_string(VK_ERROR_DEVICE_LOST), "VK_ERROR_DEVICE_LOST");
  EXPECT_EQ(vg::to_string(VK_SUCCESS), "VK_SUCCESS");
  // A non-core code still in the table resolves to its real name.
  EXPECT_EQ(vg::to_string(VK_ERROR_FRAGMENTED_POOL),
            "VK_ERROR_FRAGMENTED_POOL");
  // An unlisted code falls back to a stable, never-empty label.
  EXPECT_FALSE(vg::to_string(static_cast<VkResult>(0x7fffffff)).empty());
}

TEST(Result, HoldsValueOnSuccess) {
  vg::Result<int> result(42);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value(), 42);
  EXPECT_EQ(*result, 42);
}

TEST(Result, OperatorArrowReachesValue) {
  vg::Result<std::string> result(std::string("hello"));
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->size(), 5u);
}

TEST(Result, ConstAccessReadsValue) {
  const vg::Result<int> result(5);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value(), 5);
  EXPECT_EQ(*result, 5);
}

TEST(Result, HoldsErrorStatus) {
  vg::Result<int> result(vg::Status::error(VK_ERROR_DEVICE_LOST, "lost"));
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), VK_ERROR_DEVICE_LOST);
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

}  // namespace

TEST(Result, SupportsMoveOnlyValue) {
  vg::Result<MoveOnly> result(MoveOnly{7});
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value().value, 7);
  MoveOnly taken = std::move(result).value();  // value() && moves the value out
  EXPECT_EQ(taken.value, 7);
}

TEST(Result, OperatorStarMovesFromRvalue) {
  vg::Result<MoveOnly> result(MoveOnly{9});
  ASSERT_TRUE(result.ok());
  MoveOnly taken = *std::move(result);  // operator*() && moves the value out
  EXPECT_EQ(taken.value, 9);
}

namespace {

vg::Status try_inner(bool fail) {
  if (fail) {
    return vg::Status::error(VK_ERROR_UNKNOWN, "inner failed");
  }
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
  EXPECT_FALSE(try_outer(true).ok());
  EXPECT_EQ(try_outer(true).code(), VK_ERROR_UNKNOWN);
  EXPECT_TRUE(try_outer(false).ok());
}

TEST(VkTry, PassesSuccessAndPropagatesFailureWithExprContext) {
  EXPECT_TRUE(vk_try_caller(VK_SUCCESS).ok());

  vg::Status status = vk_try_caller(VK_ERROR_DEVICE_LOST);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), VK_ERROR_DEVICE_LOST);
  // VG_VK_TRY stringifies the expression (#expr) as the context message.
  EXPECT_NE(status.message().find("passthrough(r)"), std::string::npos);
}

namespace {

vg::Result<int> make_int(bool fail) {
  if (fail) return vg::Status::error(VK_ERROR_UNKNOWN, "no int");
  return 21;
}

// Two VG_ASSIGNs in one scope prove the __COUNTER__-keyed temporaries don't
// collide (line number is irrelevant); a failing one early-returns its Status.
vg::Result<int> sum_two(bool fail_second) {
  VG_ASSIGN(int a, make_int(false));
  VG_ASSIGN(int b, make_int(fail_second));
  return a + b;
}

// VG_ASSIGN must move the value out, so it works for move-only T.
vg::Status take_move_only() {
  VG_ASSIGN(MoveOnly m, vg::Result<MoveOnly>(MoveOnly{5}));
  if (m.value != 5) return vg::Status::error(VK_ERROR_UNKNOWN, "wrong");
  return vg::Status{};
}

}  // namespace

TEST(VgAssign, BindsValueOnSuccessAndPropagatesError) {
  vg::Result<int> ok = sum_two(/*fail_second=*/false);
  ASSERT_TRUE(ok.ok());
  EXPECT_EQ(ok.value(), 42);

  vg::Result<int> err = sum_two(/*fail_second=*/true);
  EXPECT_FALSE(err.ok());
  EXPECT_EQ(err.status().code(), VK_ERROR_UNKNOWN);
}

TEST(VgAssign, MovesMoveOnlyValue) { EXPECT_TRUE(take_move_only().ok()); }

// Reading the value of an error Result is a contract violation: VG_CHECK logs
// and aborts (SIGABRT). The "DeathTest" suffix makes gtest run this in
// isolation.
TEST(ResultDeathTest, ValueOnErrorAborts) {
  vg::Result<int> result(vg::Status::error(VK_ERROR_DEVICE_LOST, "lost"));
  EXPECT_DEATH(
      {
        vg::set_log_handler({});  // ensure the abort message reaches stderr
        (void)result.value();
      },
      "contract check failed");
}
