// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/check.hpp"

#include <cstdlib>
#include <string>

#include "volumetric_kit/gfx/core/log.hpp"
#include "volumetric_kit/gfx/core/result.hpp"

namespace volumetric_kit::gfx {

std::string_view to_string(Status::Code code) noexcept {
  switch (code) {
    case Status::Code::Ok:
      return "Ok";
    case Status::Code::InvalidArgument:
      return "InvalidArgument";
    case Status::Code::NotFound:
      return "NotFound";
    case Status::Code::Unsupported:
      return "Unsupported";
    case Status::Code::OutOfMemory:
      return "OutOfMemory";
    case Status::Code::IoError:
      return "IoError";
    case Status::Code::Vulkan:
      return "Vulkan";
  }
  return "Unknown";
}

std::string_view to_string(VkResult result) noexcept {
  switch (result) {
    case VK_SUCCESS:
      return "VK_SUCCESS";
    case VK_NOT_READY:
      return "VK_NOT_READY";
    case VK_TIMEOUT:
      return "VK_TIMEOUT";
    case VK_INCOMPLETE:
      return "VK_INCOMPLETE";
    case VK_SUBOPTIMAL_KHR:
      return "VK_SUBOPTIMAL_KHR";
    case VK_ERROR_OUT_OF_HOST_MEMORY:
      return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
      return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED:
      return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST:
      return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED:
      return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT:
      return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT:
      return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT:
      return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER:
      return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_FORMAT_NOT_SUPPORTED:
      return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case VK_ERROR_OUT_OF_POOL_MEMORY:
      return "VK_ERROR_OUT_OF_POOL_MEMORY";
    case VK_ERROR_INVALID_EXTERNAL_HANDLE:
      return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
    case VK_ERROR_SURFACE_LOST_KHR:
      return "VK_ERROR_SURFACE_LOST_KHR";
    case VK_ERROR_OUT_OF_DATE_KHR:
      return "VK_ERROR_OUT_OF_DATE_KHR";
    case VK_ERROR_UNKNOWN:
      return "VK_ERROR_UNKNOWN";
    default:
      return "VK_RESULT_UNKNOWN";
  }
}

}  // namespace volumetric_kit::gfx

namespace volumetric_kit::gfx::detail {

void check_failed(const char* file, int line, const char* expr,
                  std::string_view msg) {
  std::string text = "contract check failed: ";
  text.append(msg.data(), msg.size());
  text += " [";
  text += expr;
  text += "] at ";
  text += file;
  text += ':';
  text += std::to_string(line);
  log_message(LogLevel::Error, text);
  std::abort();
}

}  // namespace volumetric_kit::gfx::detail
