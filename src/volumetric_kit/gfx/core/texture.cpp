// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/texture.hpp"

#include <utility>

namespace volumetric_kit::gfx {

Texture::Texture(VkImage image, VkImageView view, VkExtent2D extent,
                 uint32_t depth, uint32_t mip_levels, VkFormat format,
                 std::function<void()> deleter) noexcept
    : image_(image),
      view_(view),
      extent_(extent),
      depth_(depth),
      mip_levels_(mip_levels),
      format_(format),
      deleter_(std::move(deleter)) {}

Texture::Texture(Texture&& other) noexcept
    : image_(other.image_),
      view_(other.view_),
      extent_(other.extent_),
      depth_(other.depth_),
      mip_levels_(other.mip_levels_),
      format_(other.format_),
      deleter_(std::move(other.deleter_)) {
  other.image_ = VK_NULL_HANDLE;
  other.view_ = VK_NULL_HANDLE;
  other.extent_ = {};
  other.depth_ = 1;
  other.mip_levels_ = 1;
  other.format_ = VK_FORMAT_UNDEFINED;
  other.deleter_ = nullptr;
}

Texture& Texture::operator=(Texture&& other) noexcept {
  if (this != &other) {
    destroy();
    image_ = other.image_;
    view_ = other.view_;
    extent_ = other.extent_;
    depth_ = other.depth_;
    mip_levels_ = other.mip_levels_;
    format_ = other.format_;
    deleter_ = std::move(other.deleter_);
    other.image_ = VK_NULL_HANDLE;
    other.view_ = VK_NULL_HANDLE;
    other.extent_ = {};
    other.depth_ = 1;
    other.mip_levels_ = 1;
    other.format_ = VK_FORMAT_UNDEFINED;
    other.deleter_ = nullptr;
  }
  return *this;
}

Texture::~Texture() { destroy(); }

void Texture::destroy() noexcept {
  if (deleter_) {
    deleter_();
  }
  image_ = VK_NULL_HANDLE;
  view_ = VK_NULL_HANDLE;
  extent_ = {};
  depth_ = 1;
  mip_levels_ = 1;
  format_ = VK_FORMAT_UNDEFINED;
  deleter_ = nullptr;
}

}  // namespace volumetric_kit::gfx
