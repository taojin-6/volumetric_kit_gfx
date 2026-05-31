// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/command_buffer.hpp"

namespace volumetric_kit::gfx {

CommandBuffer::CommandBuffer(VkDevice device, VkCommandPool pool,
                             VkCommandBuffer command_buffer) noexcept
    : device_(device), pool_(pool), command_buffer_(command_buffer) {}

CommandBuffer::CommandBuffer(CommandBuffer&& other) noexcept
    : device_(other.device_),
      pool_(other.pool_),
      command_buffer_(other.command_buffer_) {
  other.device_ = VK_NULL_HANDLE;
  other.pool_ = VK_NULL_HANDLE;
  other.command_buffer_ = VK_NULL_HANDLE;
}

CommandBuffer& CommandBuffer::operator=(CommandBuffer&& other) noexcept {
  if (this != &other) {
    destroy();
    device_ = other.device_;
    pool_ = other.pool_;
    command_buffer_ = other.command_buffer_;
    other.device_ = VK_NULL_HANDLE;
    other.pool_ = VK_NULL_HANDLE;
    other.command_buffer_ = VK_NULL_HANDLE;
  }
  return *this;
}

CommandBuffer::~CommandBuffer() { destroy(); }

void CommandBuffer::destroy() noexcept {
  if (command_buffer_ != VK_NULL_HANDLE) {
    vkFreeCommandBuffers(device_, pool_, 1, &command_buffer_);
  }
  device_ = VK_NULL_HANDLE;
  pool_ = VK_NULL_HANDLE;
  command_buffer_ = VK_NULL_HANDLE;
}

Status CommandBuffer::begin(VkCommandBufferUsageFlags flags) {
  VkCommandBufferBeginInfo info{};
  info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  info.flags = flags;
  VG_VK_TRY(vkBeginCommandBuffer(command_buffer_, &info));
  return Status{};
}

Status CommandBuffer::end() {
  VG_VK_TRY(vkEndCommandBuffer(command_buffer_));
  return Status{};
}

}  // namespace volumetric_kit::gfx
