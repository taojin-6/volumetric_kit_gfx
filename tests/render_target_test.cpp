// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// RenderTargetLayout compatibility + RenderTarget attachment/layout plumbing.
// These are pure value-type checks needing no device: RenderTarget is a
// non-owning view (no handle ownership), so there is no move-only lifecycle to
// exercise -- only its construction, the layout it derives, and the checks
// on a multisampled target and scope.

#include <gtest/gtest.h>

#include <cstdint>

#include "volumetric_kit/core/base/log.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"

namespace vg = volumetric_kit::gfx;
namespace vkc = volumetric_kit::core;

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

// A resolve attachment for `color`: its format, and a placeholder view no
// test here passes to Vulkan.
vg::RenderTargetAttachment resolve_of(const vg::RenderTargetAttachment& color) {
  vg::RenderTargetAttachment resolve = color;
  resolve.view = reinterpret_cast<VkImageView>(uintptr_t{2});
  return resolve;
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

// Both kinds of attachment are cleared, color is stored to be presented or read
// back, and depth, which nothing reads after the scope, is dropped.
TEST(RenderTargetBeginInfoTest, DefaultsClearAllAndStoreOnlyColor) {
  const vg::RenderTargetBeginInfo info;
  EXPECT_EQ(info.load_op, VK_ATTACHMENT_LOAD_OP_CLEAR);
  EXPECT_EQ(info.store_op, VK_ATTACHMENT_STORE_OP_STORE);
  EXPECT_EQ(info.depth_store_op, VK_ATTACHMENT_STORE_OP_DONT_CARE);
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
  vg::RenderTarget target({64, 48}, &color, 1, VK_SAMPLE_COUNT_1_BIT);

  EXPECT_TRUE(target.valid());
  EXPECT_EQ(target.extent().width, 64u);
  EXPECT_EQ(target.extent().height, 48u);
  const vg::RenderTargetLayout layout = target.layout();
  EXPECT_EQ(layout.color_count, 1u);
  EXPECT_EQ(layout.color_formats[0], kColor);
  EXPECT_EQ(layout.depth_format, VK_FORMAT_UNDEFINED);
}

// A multisampled target reports its samples and its colours' format, which
// its resolves share.
TEST(RenderTargetTest, MultisampledTargetDerivesItsLayout) {
  vg::RenderTargetAttachment color{};
  color.format = kColor;
  vg::RenderTargetAttachment depth{};
  depth.view = reinterpret_cast<VkImageView>(uintptr_t{1});
  depth.format = VK_FORMAT_D32_SFLOAT;
  const vg::RenderTargetAttachment resolve = resolve_of(color);
  const vg::RenderTarget target({64, 48}, &color, 1, vg::kWindowSamples, &depth,
                                &resolve);
  const vg::RenderTargetLayout layout = target.layout();
  EXPECT_EQ(layout.samples, vg::kWindowSamples);
  EXPECT_EQ(layout.color_formats[0], kColor);
  EXPECT_EQ(layout.depth_format, VK_FORMAT_D32_SFLOAT);
}

// What a multisampled target cannot be are programmer errors, caught before
// any Vulkan call, so no device is needed; the "DeathTest" suffix runs them
// isolated.
TEST(RenderTargetDeathTest, ResolvesGoWithMultisamplingAlone) {
  vg::RenderTargetAttachment color{};
  color.format = kColor;
  const vg::RenderTargetAttachment resolve = resolve_of(color);
  EXPECT_DEATH(
      {
        vkc::set_log_handler({});  // route the abort message to stderr
        vg::RenderTarget({8, 8}, &color, 1, vg::kWindowSamples);
      },
      "resolve attachments exactly when samples > 1");
  EXPECT_DEATH(
      {
        vkc::set_log_handler({});
        vg::RenderTarget({8, 8}, &color, 1, VK_SAMPLE_COUNT_1_BIT, nullptr,
                         &resolve);
      },
      "resolve attachments exactly when samples > 1");
}

TEST(RenderTargetDeathTest, AResolveKeepsItsColorsFormat) {
  vg::RenderTargetAttachment color{};
  color.format = kColor;
  vg::RenderTargetAttachment resolve = resolve_of(color);
  resolve.format = VK_FORMAT_B8G8R8A8_UNORM;
  EXPECT_DEATH(
      {
        vkc::set_log_handler({});
        vg::RenderTarget({8, 8}, &color, 1, vg::kWindowSamples, nullptr,
                         &resolve);
      },
      "resolve format differs");
}

// A resolve is written through its view, so it must have one.
TEST(RenderTargetDeathTest, AResolveHasAView) {
  vg::RenderTargetAttachment color{};
  color.format = kColor;
  vg::RenderTargetAttachment resolve = resolve_of(color);
  resolve.view = VK_NULL_HANDLE;
  EXPECT_DEATH(
      {
        vkc::set_log_handler({});
        vg::RenderTarget({8, 8}, &color, 1, vg::kWindowSamples, nullptr,
                         &resolve);
      },
      "resolve attachment has no view");
}

// The samples never outlive a multisampled scope: nothing loads them, and
// depth is never stored.
TEST(RenderTargetDeathTest, AMultisampledScopeNeitherLoadsNorStoresDepth) {
  vg::RenderTargetAttachment color{};
  color.format = kColor;
  const vg::RenderTargetAttachment resolve = resolve_of(color);
  const vg::RenderTarget target({8, 8}, &color, 1, vg::kWindowSamples, nullptr,
                                &resolve);
  vg::RenderTargetBeginInfo load;
  load.load_op = VK_ATTACHMENT_LOAD_OP_LOAD;
  EXPECT_DEATH(
      {
        vkc::set_log_handler({});
        target.begin(VK_NULL_HANDLE, load);
      },
      "cannot LOAD");
  vg::RenderTargetBeginInfo store_depth;
  store_depth.depth_store_op = VK_ATTACHMENT_STORE_OP_STORE;
  EXPECT_DEATH(
      {
        vkc::set_log_handler({});
        target.begin(VK_NULL_HANDLE, store_depth);
      },
      "cannot STORE depth");
}

}  // namespace
