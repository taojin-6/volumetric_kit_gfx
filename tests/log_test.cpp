// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <vector>

#include "volumetric_kit/gfx/core/log.hpp"

namespace vg = volumetric_kit::gfx;

namespace {

// Restores the default sink after each test so an installed handler never leaks
// into another test (or a death test that expects the default stderr sink).
class LogTest : public ::testing::Test {
 protected:
  void TearDown() override { vg::set_log_handler({}); }
};

}  // namespace

// gfx's diagnostics reach the family's one sink with source "vg", so the
// default sink keeps printing [vg <level>].
TEST_F(LogTest, HandlerReceivesLevelSourceAndMessage) {
  vg::LogLevel level = vg::LogLevel::Debug;
  std::string source;
  std::string message;
  int calls = 0;
  vg::set_log_handler(
      [&](vg::LogLevel l, std::string_view src, std::string_view m) {
        level = l;
        source.assign(src);
        message.assign(m);
        ++calls;
      });

  vg::log_message(vg::LogLevel::Error, "boom");

  EXPECT_EQ(calls, 1);
  EXPECT_EQ(level, vg::LogLevel::Error);
  EXPECT_EQ(source, "vg");
  EXPECT_EQ(message, "boom");
}

TEST_F(LogTest, DefaultSinkPrintsTheVgSource) {
  testing::internal::CaptureStderr();
  vg::log_message(vg::LogLevel::Warning, "careful");
  EXPECT_EQ(testing::internal::GetCapturedStderr(), "[vg warning] careful\n");
}

TEST_F(LogTest, AllLevelsRouteToHandler) {
  // Filtering by severity is the default sink's job; an installed handler sees
  // every level so consumers can route them as they like.
  std::vector<vg::LogLevel> seen;
  vg::set_log_handler([&](vg::LogLevel l, std::string_view, std::string_view) {
    seen.push_back(l);
  });

  vg::log_message(vg::LogLevel::Debug, "d");
  vg::log_message(vg::LogLevel::Info, "i");
  vg::log_message(vg::LogLevel::Warning, "w");
  vg::log_message(vg::LogLevel::Error, "e");

  ASSERT_EQ(seen.size(), 4u);
  EXPECT_EQ(seen[0], vg::LogLevel::Debug);
  EXPECT_EQ(seen[3], vg::LogLevel::Error);
}

TEST_F(LogTest, ResettingHandlerStopsDelivery) {
  int calls = 0;
  vg::set_log_handler(
      [&](vg::LogLevel, std::string_view, std::string_view) { ++calls; });
  vg::log_message(vg::LogLevel::Error, "x");
  ASSERT_EQ(calls, 1);

  vg::set_log_handler({});  // back to the built-in stderr sink
  vg::log_message(vg::LogLevel::Error, "y");  // must not reach our handler
  EXPECT_EQ(calls, 1);
}
