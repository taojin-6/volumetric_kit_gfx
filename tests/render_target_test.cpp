// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// RenderTargetLayout compatibility + RenderTarget attachment/layout plumbing.
// These are pure value-type checks needing no device: RenderTarget is a
// non-owning view (no handle ownership), so there is no move-only lifecycle to
// exercise -- only its construction and the layout it derives.

#include <gtest/gtest.h>

#include "volumetric_kit/gfx/core/render_target.hpp"

namespace vg = volumetric_kit::gfx;

namespace {

constexpr VkFormat kColor = VK_FORMAT_R8G8B8A8_UNORM;

vg::RenderTargetLayout color_layout(
    VkFormat color, VkFormat depth = VK_FORMAT_UNDEFINED,
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT) {
  vg::RenderTargetLayout layout;
  layout.color_formats[0] = color;
  layout.color_count = 1;
  layout.depth_format = depth;
  layout.samples = samples;
  return layout;
}

TEST(RenderTargetLayoutTest, IdenticalLayoutsAreCompatible) {
  EXPECT_TRUE(color_layout(kColor).compatible_with(color_layout(kColor)));
}

TEST(RenderTargetLayoutTest, DifferentColorFormatIsIncompatible) {
  EXPECT_FALSE(color_layout(kColor).compatible_with(
      color_layout(VK_FORMAT_B8G8R8A8_UNORM)));
}

TEST(RenderTargetLayoutTest, DifferentColorCountIsIncompatible) {
  vg::RenderTargetLayout two = color_layout(kColor);
  two.color_formats[1] = kColor;
  two.color_count = 2;
  EXPECT_FALSE(color_layout(kColor).compatible_with(two));
}

TEST(RenderTargetLayoutTest, DifferentDepthFormatIsIncompatible) {
  EXPECT_FALSE(color_layout(kColor).compatible_with(
      color_layout(kColor, VK_FORMAT_D32_SFLOAT)));
}

TEST(RenderTargetLayoutTest, DifferentSampleCountIsIncompatible) {
  EXPECT_FALSE(color_layout(kColor).compatible_with(
      color_layout(kColor, VK_FORMAT_UNDEFINED, VK_SAMPLE_COUNT_4_BIT)));
}

TEST(RenderTargetLayoutTest, FormatsPastColorCountAreIgnored) {
  // Only the first color_count formats are significant; trailing junk must not
  // affect compatibility.
  vg::RenderTargetLayout other = color_layout(kColor);
  other.color_formats[1] = VK_FORMAT_B8G8R8A8_UNORM;  // beyond color_count == 1
  EXPECT_TRUE(color_layout(kColor).compatible_with(other));
}

TEST(RenderTargetTest, DefaultConstructedIsEmpty) {
  vg::RenderTarget target;
  EXPECT_FALSE(target.valid());
  EXPECT_EQ(target.layout().color_count, 0u);
}

TEST(RenderTargetTest, DerivesLayoutAndExtentFromAttachments) {
  // The handles are not dereferenced here (no begin/end), so only the format
  // matters to the derived layout.
  vg::RenderTargetAttachment color{};
  color.format = kColor;
  vg::RenderTarget target({64, 48}, &color, 1, nullptr, VK_SAMPLE_COUNT_1_BIT);

  EXPECT_TRUE(target.valid());
  EXPECT_EQ(target.extent().width, 64u);
  EXPECT_EQ(target.extent().height, 48u);
  const vg::RenderTargetLayout layout = target.layout();
  EXPECT_EQ(layout.color_count, 1u);
  EXPECT_EQ(layout.color_formats[0], kColor);
  EXPECT_EQ(layout.depth_format, VK_FORMAT_UNDEFINED);
}

TEST(RenderTargetTest, DepthAttachmentShowsInDerivedLayout) {
  vg::RenderTargetAttachment color{};
  color.format = kColor;
  vg::RenderTargetAttachment depth{};
  depth.format = VK_FORMAT_D32_SFLOAT;
  vg::RenderTarget target({16, 16}, &color, 1, &depth, VK_SAMPLE_COUNT_1_BIT);

  EXPECT_EQ(target.layout().depth_format, VK_FORMAT_D32_SFLOAT);
}

}  // namespace
