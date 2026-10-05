// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <vector>

#include "volumetric_kit/gfx/core/log.hpp"

namespace vg = volumetric_kit::gfx;
namespace vkc = volumetric_kit::core;

namespace {

// Restores the default sink after each test so an installed handler never leaks
// into another test (or a death test that expects the default stderr sink).
class LogTest : public ::testing::Test {
 protected:
  void TearDown() override { vkc::set_log_handler({}); }
};

}  // namespace

// gfx's diagnostics reach the family's one sink with source "vg", so the
// default sink keeps printing [vg <level>].
TEST_F(LogTest, HandlerReceivesLevelSourceAndMessage) {
  vkc::LogLevel level = vkc::LogLevel::Debug;
  std::string source;
  std::string message;
  int calls = 0;
  vkc::set_log_handler(
      [&](vkc::LogLevel l, std::string_view src, std::string_view m) {
        level = l;
        source.assign(src);
        message.assign(m);
        ++calls;
      });

  vg::log_message(vkc::LogLevel::Error, "boom");

  EXPECT_EQ(calls, 1);
  EXPECT_EQ(level, vkc::LogLevel::Error);
  EXPECT_EQ(source, "vg");
  EXPECT_EQ(message, "boom");
}

TEST_F(LogTest, DefaultSinkPrintsTheVgSource) {
  testing::internal::CaptureStderr();
  vg::log_message(vkc::LogLevel::Warning, "careful");
  EXPECT_EQ(testing::internal::GetCapturedStderr(), "[vg warning] careful\n");
}

TEST_F(LogTest, AllLevelsRouteToHandler) {
  // Filtering by severity is the default sink's job; an installed handler sees
  // every level so consumers can route them as they like.
  std::vector<vkc::LogLevel> seen;
  vkc::set_log_handler([&](vkc::LogLevel l, std::string_view,
                           std::string_view) { seen.push_back(l); });

  vg::log_message(vkc::LogLevel::Debug, "d");
  vg::log_message(vkc::LogLevel::Info, "i");
  vg::log_message(vkc::LogLevel::Warning, "w");
  vg::log_message(vkc::LogLevel::Error, "e");

  ASSERT_EQ(seen.size(), 4u);
  EXPECT_EQ(seen[0], vkc::LogLevel::Debug);
  EXPECT_EQ(seen[3], vkc::LogLevel::Error);
}

TEST_F(LogTest, ResettingHandlerStopsDelivery) {
  int calls = 0;
  vkc::set_log_handler(
      [&](vkc::LogLevel, std::string_view, std::string_view) { ++calls; });
  vg::log_message(vkc::LogLevel::Error, "x");
  ASSERT_EQ(calls, 1);

  vkc::set_log_handler({});  // back to the built-in stderr sink
  vg::log_message(vkc::LogLevel::Error, "y");  // must not reach our handler
  EXPECT_EQ(calls, 1);
}
