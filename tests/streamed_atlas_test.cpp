// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// StreamedAtlas: a HybridMeshPipeline atlas updated inside frames. A frame
// here is a command buffer the test submits itself, setting the frame's number
// on a timeline the test owns -- as windowing::FrameLoop's frames set theirs,
// recorded with core::note_timeline_signals as the loop records them. A gate
// semaphore the test sets from the host holds frames in flight. Every frame
// draws a full-screen quad whose uv0 put target pixel (x, y) on atlas texel
// (x, y), so its readback shows the picture it sampled. Runs under the
// validation layer with synchronization validation, tracking what shaders
// read through descriptors.

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/command_buffer.hpp"
#include "volumetric_kit/core/vulkan/command_pool.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/core/vulkan/sync.hpp"
#include "volumetric_kit/gfx/assets/mesh.hpp"
#include "volumetric_kit/gfx/core/image_update.hpp"
#include "volumetric_kit/gfx/core/offscreen_target.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/core/sampler.hpp"
#include "volumetric_kit/gfx/core/texture_upload.hpp"
#include "volumetric_kit/gfx/pipelines/gpu_mesh.hpp"
#include "volumetric_kit/gfx/pipelines/hybrid_mesh_pipeline.hpp"
#include "volumetric_kit/gfx/pipelines/streamed_atlas.hpp"
#include "vulkan_test_fixture.hpp"

namespace {

namespace pipelines = volumetric_kit::gfx::pipelines;
namespace assets = volumetric_kit::gfx::assets;

// The atlas and the target are both kSide x kSide texels.
constexpr uint32_t kSide = 2;
constexpr VkFormat kFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;

using Rgba = std::array<uint8_t, 4>;
using Picture = std::array<Rgba, kSide * kSide>;  // row-major texels

constexpr Rgba kRed{255, 0, 0, 255};
constexpr Rgba kGreen{0, 255, 0, 255};
constexpr Rgba kBlue{0, 0, 255, 255};
constexpr Rgba kWhite{255, 255, 255, 255};
constexpr Rgba kYellow{255, 255, 0, 255};
constexpr Rgba kMagenta{255, 0, 255, 255};
constexpr Rgba kCyan{0, 255, 255, 255};  // the quad's vertex color

Picture solid(const Rgba& color) {
  Picture picture;
  picture.fill(color);
  return picture;
}

vg::RenderTargetLayout target_layout() {
  vg::RenderTargetLayout layout;
  layout.color_formats[0] = kFormat;
  layout.color_count = 1;
  layout.depth_format = kDepthFormat;
  return layout;
}

// A quad over the whole target, uv0 its own position in [0, 1]^2, so each
// pixel centre samples the centre of the texel it lies over; cyan vertices.
assets::Mesh make_quad() {
  auto vert = [](float x, float y) {
    assets::Vertex v;
    v.position = {x, y, 0.5f};
    v.normal = {0.0f, 0.0f, 1.0f};
    v.uv0 = {(x + 1.0f) * 0.5f, (y + 1.0f) * 0.5f};
    v.color = {0.0f, 1.0f, 1.0f, 1.0f};
    return v;
  };
  assets::Mesh mesh;
  mesh.vertices = {vert(-1.0f, -1.0f), vert(1.0f, -1.0f), vert(1.0f, 1.0f),
                   vert(-1.0f, 1.0f)};
  mesh.indices = {0, 2, 1, 0, 3, 2};
  return mesh;
}

// Sets the gate, letting every frame held at it run, unless it is set.
void open_gate(vkc::TimelineSemaphore& gate) {
  const vkc::Result<uint64_t> value = gate.value();
  if (value.ok() && value.value() < 1) {
    EXPECT_TRUE(gate.signal(1).ok());
  }
}

// Opens the gate when it goes out of scope, so a test that stops early never
// leaves frames held: declare it after whatever waits for those frames.
class OpenGateAtExit {
 public:
  explicit OpenGateAtExit(vkc::TimelineSemaphore& gate) : gate_(gate) {}
  ~OpenGateAtExit() { open_gate(gate_); }
  OpenGateAtExit(const OpenGateAtExit&) = delete;
  OpenGateAtExit& operator=(const OpenGateAtExit&) = delete;

 private:
  vkc::TimelineSemaphore& gate_;
};

// Opens the gate from another thread after a pause, so this one can block on
// a frame held at it.
class OpenGateLater {
 public:
  explicit OpenGateLater(vkc::TimelineSemaphore& gate)
      : thread_([&gate, this] {
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
          opened_ = true;
          open_gate(gate);
        }) {}
  ~OpenGateLater() { thread_.join(); }
  OpenGateLater(const OpenGateLater&) = delete;
  OpenGateLater& operator=(const OpenGateLater&) = delete;

  bool opened() const { return opened_; }

 private:
  std::atomic<bool> opened_{false};
  std::thread thread_;
};

// --- No device -------------------------------------------------------------

TEST(StreamedAtlasEmptyTest, DefaultConstructedIsEmpty) {
  pipelines::StreamedAtlas atlas;
  EXPECT_FALSE(atlas.valid());
  EXPECT_FALSE(atlas.has_picture());
  EXPECT_EQ(atlas.picture(), nullptr);
  EXPECT_EQ(atlas.slot_count(), 0u);
  EXPECT_EQ(atlas.extent().width, 0u);
  EXPECT_EQ(atlas.format(), VK_FORMAT_UNDEFINED);
  EXPECT_EQ(atlas.use(1), VK_NULL_HANDLE);
  const Picture picture = solid(kRed);
  const vkc::Status refused =
      atlas.record_upload(VK_NULL_HANDLE, 1, picture.data(), sizeof(picture));
  EXPECT_EQ(refused.domain(), vkc::Status::Code::InvalidArgument);
}

// --- On a device: frames under synchronization validation -------------------

class StreamedAtlasTest : public VulkanDeviceTest {
 protected:
  bool wants_validation() const override { return true; }
  bool wants_sync_validation() const override { return true; }
  // The draws sample the atlas through a descriptor, which synchronization
  // validation tracks only with this.
  bool wants_shader_access_validation() const override { return true; }

  // One frame: its number, its command buffer and the target it draws into.
  struct Frame {
    uint64_t number = 0;
    vkc::CommandBuffer cmd;
    vg::OffscreenTarget target;
  };

  void SetUp() override {
    VulkanDeviceTest::SetUp();
    if (base_setup_incomplete()) {
      return;
    }
    auto allocator = vkc::Allocator::create(instance_->handle(), *device_);
    ASSERT_TRUE(allocator.ok()) << allocator.status().message();
    allocator_.emplace(std::move(allocator).value());
    auto pipeline = pipelines::HybridMeshPipeline::create(*device_, *allocator_,
                                                          target_layout());
    ASSERT_TRUE(pipeline.ok()) << pipeline.status().message();
    pipeline_.emplace(std::move(pipeline).value());
    auto mesh = pipelines::upload_mesh(*device_, *allocator_, make_quad());
    ASSERT_TRUE(mesh.ok()) << mesh.status().message();
    mesh_.emplace(std::move(mesh).value());
    auto timeline = vkc::TimelineSemaphore::create(*device_, 0);
    ASSERT_TRUE(timeline.ok()) << timeline.status().message();
    timeline_.emplace(std::move(timeline).value());
    auto gate = vkc::TimelineSemaphore::create(*device_, 0);
    ASSERT_TRUE(gate.ok()) << gate.status().message();
    gate_.emplace(std::move(gate).value());
    auto pool = vkc::CommandPool::create(device(), device_->queue_family());
    ASSERT_TRUE(pool.ok()) << pool.status().message();
    pool_.emplace(std::move(pool).value());
  }

  void TearDown() override {
    // No frame may stay held at the gate while the base waits for the
    // device to go idle.
    if (gate_) {
      open_gate(*gate_);
    }
    VulkanDeviceTest::TearDown();
  }

  vkc::Result<pipelines::StreamedAtlas> make_atlas(uint32_t slots) {
    pipelines::StreamedAtlasDesc desc;
    desc.extent = {kSide, kSide};
    desc.format = kFormat;
    desc.slots = slots;
    return pipelines::StreamedAtlas::create(*pipeline_, *allocator_, *timeline_,
                                            desc);
  }

  // Begin frame `number`: a command buffer recording, and a target. Null on
  // a failure, which has been reported.
  Frame* begin_frame(uint64_t number) {
    auto frame = std::make_unique<Frame>();
    frame->number = number;
    auto cmd = pool_->allocate_primary();
    EXPECT_TRUE(cmd.ok()) << cmd.status().message();
    if (!cmd.ok()) {
      return nullptr;
    }
    frame->cmd = std::move(cmd).value();
    EXPECT_TRUE(
        frame->cmd.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT).ok());
    auto target = make_target();
    EXPECT_TRUE(target.ok()) << target.status().message();
    if (!target.ok()) {
      return nullptr;
    }
    frame->target = std::move(target).value();
    frames_.push_back(std::move(frame));
    return frames_.back().get();
  }

  // Draw the quad into `target` with `atlas` as set 0, then record the
  // target's readback.
  void draw(VkCommandBuffer cmd, const vg::OffscreenTarget& target,
            VkDescriptorSet atlas) {
    target.prepare(cmd);
    vg::RenderTargetBeginInfo begin;
    begin.clear_color.float32[3] = 1.0f;
    const vg::RenderTarget rt = target.target();
    rt.begin(cmd, begin);
    const pipelines::HybridMeshDraw mesh{&*mesh_};
    pipelines::HybridMeshFrame hybrid;
    hybrid.extent = {kSide, kSide};
    hybrid.flags = 0;  // unlit: the albedo itself
    hybrid.atlas = atlas;
    hybrid.draws = &mesh;
    hybrid.draw_count = 1;
    pipeline_->submit(cmd, hybrid);
    rt.end(cmd);
    target.record_readback(cmd);
  }

  // Draw the quad into the frame's own target.
  void draw(Frame& frame, VkDescriptorSet atlas) {
    draw(frame.cmd.handle(), frame.target, atlas);
  }

  vkc::Result<vg::OffscreenTarget> make_target() {
    vg::OffscreenTargetDesc desc;
    desc.extent = {kSide, kSide};
    desc.color_format = kFormat;
    desc.depth_format = kDepthFormat;
    return vg::OffscreenTarget::create(*allocator_, desc);
  }

  // End and submit the frame, which sets its number on the timeline when it
  // completes; a `gated` one is held until the gate opens.
  void submit(Frame& frame, bool gated = false) {
    ASSERT_TRUE(frame.cmd.end().ok());
    const VkCommandBuffer cmd = frame.cmd.handle();
    const uint64_t open = 1;
    const VkSemaphore gate = gate_->handle();
    const VkPipelineStageFlags held = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    const VkSemaphore timeline = timeline_->handle();
    VkTimelineSemaphoreSubmitInfo values{};
    values.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    values.waitSemaphoreValueCount = gated ? 1 : 0;
    values.pWaitSemaphoreValues = &open;
    values.signalSemaphoreValueCount = 1;
    values.pSignalSemaphoreValues = &frame.number;
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.pNext = &values;
    submit.waitSemaphoreCount = gated ? 1 : 0;
    submit.pWaitSemaphores = &gate;
    submit.pWaitDstStageMask = &held;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &timeline;
    ASSERT_EQ(device_->queue_submit(1, &submit, VK_NULL_HANDLE), VK_SUCCESS);
    vkc::note_timeline_signals({{&*timeline_, frame.number}});
  }

  // What the frame drew, once it completed.
  Picture drawn(const Frame& frame) {
    Picture picture{};
    EXPECT_TRUE(timeline_->wait(frame.number).ok());
    const auto* px = static_cast<const uint8_t*>(frame.target.pixels());
    EXPECT_NE(px, nullptr);
    if (px != nullptr) {
      std::memcpy(picture.data(), px, sizeof(picture));
    }
    return picture;
  }

  // Frame `number` uploads `picture` into `atlas` and draws it, ungated.
  // Returns the frame, or null on a failure, which has been reported.
  Frame* upload_and_draw(pipelines::StreamedAtlas& atlas, uint64_t number,
                         const Picture& picture, bool gated = false) {
    Frame* frame = begin_frame(number);
    if (frame == nullptr) {
      return nullptr;
    }
    const vkc::Status uploaded = atlas.record_upload(
        frame->cmd.handle(), number, picture.data(), sizeof(picture));
    EXPECT_TRUE(uploaded.ok()) << uploaded.message();
    draw(*frame, atlas.use(number));
    submit(*frame, gated);
    return frame;
  }

  std::optional<vkc::Allocator> allocator_;
  std::optional<pipelines::HybridMeshPipeline> pipeline_;
  std::optional<pipelines::GpuMesh> mesh_;
  std::optional<vkc::TimelineSemaphore> timeline_;
  std::optional<vkc::TimelineSemaphore> gate_;
  std::optional<vkc::CommandPool> pool_;
  // Destroyed after the base TearDown has waited for the device.
  std::vector<std::unique_ptr<Frame>> frames_;
};

TEST_F(StreamedAtlasTest, CreateRefusesWhatItCannotHold) {
  pipelines::StreamedAtlasDesc desc;
  desc.extent = {kSide, kSide};
  desc.format = kFormat;

  const pipelines::HybridMeshPipeline empty_pipeline;
  EXPECT_EQ(pipelines::StreamedAtlas::create(empty_pipeline, *allocator_,
                                             *timeline_, desc)
                .status()
                .domain(),
            vkc::Status::Code::InvalidArgument);
  const vkc::TimelineSemaphore empty_timeline;
  EXPECT_EQ(pipelines::StreamedAtlas::create(*pipeline_, *allocator_,
                                             empty_timeline, desc)
                .status()
                .domain(),
            vkc::Status::Code::InvalidArgument);

  pipelines::StreamedAtlasDesc bad = desc;
  bad.extent = {0, kSide};
  EXPECT_EQ(
      pipelines::StreamedAtlas::create(*pipeline_, *allocator_, *timeline_, bad)
          .status()
          .domain(),
      vkc::Status::Code::InvalidArgument);
  bad = desc;
  bad.slots = 0;
  EXPECT_EQ(
      pipelines::StreamedAtlas::create(*pipeline_, *allocator_, *timeline_, bad)
          .status()
          .domain(),
      vkc::Status::Code::InvalidArgument);
  bad = desc;
  bad.extent = {caps_.limits().maxImageDimension2D + 1, kSide};
  EXPECT_EQ(
      pipelines::StreamedAtlas::create(*pipeline_, *allocator_, *timeline_, bad)
          .status()
          .domain(),
      vkc::Status::Code::InvalidArgument);
  bad = desc;
  bad.format = VK_FORMAT_D32_SFLOAT;  // not a color format
  EXPECT_EQ(
      pipelines::StreamedAtlas::create(*pipeline_, *allocator_, *timeline_, bad)
          .status()
          .domain(),
      vkc::Status::Code::Unsupported);
}

// Before its first update the atlas has no picture, and the pipeline draws in
// vertex color; the frame that uploads one draws it, every texel where it
// belongs.
TEST_F(StreamedAtlasTest,
       DrawsInVertexColorUntilTheFirstPictureThenThePicture) {
  auto atlas = make_atlas(3);
  ASSERT_TRUE(atlas.ok()) << atlas.status().message();
  EXPECT_FALSE(atlas.value().has_picture());

  Frame* first = begin_frame(1);
  ASSERT_NE(first, nullptr);
  const VkDescriptorSet none = atlas.value().use(1);
  EXPECT_EQ(none, VK_NULL_HANDLE);
  draw(*first, none);
  submit(*first);
  EXPECT_EQ(drawn(*first), solid(kCyan));

  const Picture picture{kRed, kGreen, kBlue, kWhite};
  Frame* second = upload_and_draw(atlas.value(), 2, picture);
  ASSERT_NE(second, nullptr);
  EXPECT_TRUE(atlas.value().has_picture());
  ASSERT_NE(atlas.value().picture(), nullptr);
  EXPECT_EQ(atlas.value().picture()->layout(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  EXPECT_EQ(drawn(*second), picture);
}

// An update from rows of a device buffer: two one-texel-wide tiles, each a
// column of the picture, read from a buffer with a row length and offsets of
// its own, land where their regions put them.
TEST_F(StreamedAtlasTest, UpdateCopiesTilesFromADeviceBuffer) {
  auto atlas = make_atlas(3);
  ASSERT_TRUE(atlas.ok()) << atlas.status().message();

  // Rows of three texels; the left tile is column 0, the right column 2.
  const std::array<Rgba, 6> rows{kRed,  kMagenta, kGreen,
                                 kBlue, kMagenta, kYellow};
  vkc::BufferDesc desc;
  desc.size = sizeof(rows);
  desc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  desc.memory = vkc::MemoryUsage::Staging;
  auto source = allocator_->create_buffer(desc);
  ASSERT_TRUE(source.ok()) << source.status().message();
  std::memcpy(source.value().mapped(), rows.data(), sizeof(rows));

  std::array<VkBufferImageCopy, 2> tiles{};
  for (uint32_t t = 0; t < 2; ++t) {
    tiles[t].bufferOffset = VkDeviceSize{t} * 2 * sizeof(Rgba);
    tiles[t].bufferRowLength = 3;
    tiles[t].imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    tiles[t].imageOffset = {static_cast<int32_t>(t), 0, 0};
    tiles[t].imageExtent = {1, kSide, 1};
  }

  Frame* frame = begin_frame(1);
  ASSERT_NE(frame, nullptr);
  const vkc::Status updated = atlas.value().record_update(
      frame->cmd.handle(), 1, source.value(), tiles.data(), 2);
  EXPECT_TRUE(updated.ok()) << updated.message();
  draw(*frame, atlas.value().use(1));
  submit(*frame);
  EXPECT_EQ(drawn(*frame), (Picture{kRed, kGreen, kBlue, kYellow}));
}

// The ring's promise: while frames 1-3 are held in flight, each drawing its
// own picture, frame 4's update finds every image in use. It waits for frame
// 1 -- the oldest -- and only then takes frame 1's image, so every frame still
// draws the picture it was given.
TEST_F(StreamedAtlasTest, RingNeverOverwritesAPictureAFrameInFlightDraws) {
  auto atlas = make_atlas(3);
  ASSERT_TRUE(atlas.ok()) << atlas.status().message();
  OpenGateAtExit open_at_exit(*gate_);

  const std::array<Rgba, 4> colors{kRed, kGreen, kBlue, kYellow};
  std::array<Frame*, 4> frames{};
  std::array<VkImage, 4> images{};
  for (uint64_t n = 1; n <= 3; ++n) {
    frames[n - 1] =
        upload_and_draw(atlas.value(), n, solid(colors[n - 1]), true);
    ASSERT_NE(frames[n - 1], nullptr);
    images[n - 1] = atlas.value().picture()->handle();
  }
  EXPECT_NE(images[0], images[1]);
  EXPECT_NE(images[0], images[2]);
  EXPECT_NE(images[1], images[2]);
  ASSERT_EQ(timeline_->value().value(), 0u) << "frames 1-3 are held";

  {
    OpenGateLater open_later(*gate_);
    frames[3] = upload_and_draw(atlas.value(), 4, solid(colors[3]));
    ASSERT_NE(frames[3], nullptr);
    EXPECT_TRUE(open_later.opened()) << "the update waited for frame 1";
    EXPECT_GE(timeline_->value().value(), 1u);
  }
  images[3] = atlas.value().picture()->handle();
  EXPECT_EQ(images[3], images[0]) << "frame 4 reuses frame 1's image";

  for (size_t f = 0; f < 4; ++f) {
    EXPECT_EQ(drawn(*frames[f]), solid(colors[f])) << "frame " << f + 1;
  }
}

// Frames that complete free their images: with two images, frames 1-4
// alternate between them, each reused once its last frame has completed. The
// host learns that from the timeline's counter alone, as a frame loop's caller
// does, and no hazard is reported.
TEST_F(StreamedAtlasTest, ReusesAnImageOnceItsFrameCompletes) {
  auto atlas = make_atlas(2);
  ASSERT_TRUE(atlas.ok()) << atlas.status().message();

  const std::array<Rgba, 4> colors{kRed, kGreen, kBlue, kYellow};
  std::array<Frame*, 4> frames{};
  std::array<VkImage, 4> images{};
  for (uint64_t n = 1; n <= 4; ++n) {
    if (n > 2) {
      // The image's last frame has completed.
      while (timeline_->value().value() < n - 2) {
        std::this_thread::yield();
      }
    }
    frames[n - 1] = upload_and_draw(atlas.value(), n, solid(colors[n - 1]));
    ASSERT_NE(frames[n - 1], nullptr);
    images[n - 1] = atlas.value().picture()->handle();
  }
  EXPECT_NE(images[0], images[1]);
  EXPECT_EQ(images[2], images[0]);
  EXPECT_EQ(images[3], images[1]);
  for (size_t f = 0; f < 4; ++f) {
    EXPECT_EQ(drawn(*frames[f]), solid(colors[f])) << "frame " << f + 1;
  }
}

// A frame that has used every image cannot update again: the update is
// refused, nothing is recorded, and the frame draws the picture it had.
TEST_F(StreamedAtlasTest, RefusesAnUpdateWhenThisFrameUsesEveryImage) {
  auto atlas = make_atlas(2);
  ASSERT_TRUE(atlas.ok()) << atlas.status().message();

  Frame* frame = begin_frame(1);
  ASSERT_NE(frame, nullptr);
  const VkCommandBuffer cmd = frame->cmd.handle();
  const Picture red = solid(kRed);
  const Picture green = solid(kGreen);
  const Picture blue = solid(kBlue);
  EXPECT_TRUE(
      atlas.value().record_upload(cmd, 1, red.data(), sizeof(red)).ok());
  EXPECT_TRUE(
      atlas.value().record_upload(cmd, 1, green.data(), sizeof(green)).ok());
  const VkImage before = atlas.value().picture()->handle();
  const vkc::Status refused =
      atlas.value().record_upload(cmd, 1, blue.data(), sizeof(blue));
  EXPECT_EQ(refused.domain(), vkc::Status::Code::InvalidArgument)
      << refused.message();
  EXPECT_EQ(atlas.value().picture()->handle(), before);
  draw(*frame, atlas.value().use(1));
  submit(*frame);
  EXPECT_EQ(drawn(*frame), green);
}

// Frame numbers start at 1 and never go back.
TEST_F(StreamedAtlasTest, RefusesAFrameNumberBelowOneGiven) {
  auto atlas = make_atlas(3);
  ASSERT_TRUE(atlas.ok()) << atlas.status().message();
  const Picture red = solid(kRed);

  Frame* frame = begin_frame(5);
  ASSERT_NE(frame, nullptr);
  const VkCommandBuffer cmd = frame->cmd.handle();
  EXPECT_EQ(
      atlas.value().record_upload(cmd, 0, red.data(), sizeof(red)).domain(),
      vkc::Status::Code::InvalidArgument);
  EXPECT_TRUE(
      atlas.value().record_upload(cmd, 5, red.data(), sizeof(red)).ok());
  EXPECT_EQ(
      atlas.value().record_upload(cmd, 4, red.data(), sizeof(red)).domain(),
      vkc::Status::Code::InvalidArgument);
  // A bad picture is refused too, with the current one kept.
  EXPECT_EQ(atlas.value().record_upload(cmd, 5, red.data(), 4).domain(),
            vkc::Status::Code::InvalidArgument);
  draw(*frame, atlas.value().use(5));
  submit(*frame);
  EXPECT_EQ(drawn(*frame), red);
}

// An update must not wait for a frame that never reached the queue: it is
// refused rather than blocking forever. Destroying the atlas waits for the
// queues instead of that frame's number.
TEST_F(StreamedAtlasTest, NeverWaitsForAFrameThatWasNotSubmitted) {
  std::optional<pipelines::StreamedAtlas> atlas;
  {
    auto made = make_atlas(1);
    ASSERT_TRUE(made.ok()) << made.status().message();
    atlas.emplace(std::move(made).value());
  }
  const Picture red = solid(kRed);
  Frame* dropped = begin_frame(1);
  ASSERT_NE(dropped, nullptr);
  EXPECT_TRUE(
      atlas->record_upload(dropped->cmd.handle(), 1, red.data(), sizeof(red))
          .ok());
  // Frame 1 is never submitted.
  Frame* next = begin_frame(2);
  ASSERT_NE(next, nullptr);
  EXPECT_EQ(atlas->record_upload(next->cmd.handle(), 2, red.data(), sizeof(red))
                .domain(),
            vkc::Status::Code::InvalidArgument);
  atlas.reset();  // returns: frame 1 is not waited for
}

// Destroying an atlas waits for the newest frame that used it, so a frame
// held in flight still draws from live images.
TEST_F(StreamedAtlasTest, DestructionWaitsForTheNewestFrame) {
  std::optional<pipelines::StreamedAtlas> atlas;
  {
    auto made = make_atlas(2);
    ASSERT_TRUE(made.ok()) << made.status().message();
    atlas.emplace(std::move(made).value());
  }
  OpenGateAtExit open_at_exit(*gate_);
  Frame* frame = upload_and_draw(*atlas, 1, solid(kMagenta), true);
  ASSERT_NE(frame, nullptr);
  {
    OpenGateLater open_later(*gate_);
    atlas.reset();
    EXPECT_TRUE(open_later.opened()) << "destruction waited for frame 1";
    EXPECT_GE(timeline_->value().value(), 1u);
  }
  EXPECT_EQ(drawn(*frame), solid(kMagenta));
}

TEST_F(StreamedAtlasTest, MoveConstructLeavesTheSourceEmpty) {
  auto made = make_atlas(2);
  ASSERT_TRUE(made.ok()) << made.status().message();
  pipelines::StreamedAtlas source = std::move(made).value();
  Frame* frame = upload_and_draw(source, 1, solid(kRed));
  ASSERT_NE(frame, nullptr);
  const VkImage picture = source.picture()->handle();

  pipelines::StreamedAtlas moved(std::move(source));
  EXPECT_TRUE(moved.valid());
  ASSERT_TRUE(moved.has_picture());
  EXPECT_EQ(moved.picture()->handle(), picture);
  EXPECT_EQ(moved.slot_count(), 2u);
  // NOLINTBEGIN(bugprone-use-after-move)
  EXPECT_FALSE(source.valid());
  EXPECT_FALSE(source.has_picture());
  EXPECT_EQ(source.picture(), nullptr);
  EXPECT_EQ(source.slot_count(), 0u);
  EXPECT_EQ(source.extent().width, 0u);
  EXPECT_EQ(source.format(), VK_FORMAT_UNDEFINED);
  EXPECT_EQ(source.use(2), VK_NULL_HANDLE);
  // NOLINTEND(bugprone-use-after-move)
  EXPECT_EQ(drawn(*frame), solid(kRed));
}

// Move-assigning over a live atlas waits for the frames that used it, frees
// it, and adopts the source; the adopted atlas keeps working.
TEST_F(StreamedAtlasTest, MoveAssignOverALiveAtlasAdoptsTheSource) {
  auto first = make_atlas(2);
  ASSERT_TRUE(first.ok()) << first.status().message();
  auto second = make_atlas(3);
  ASSERT_TRUE(second.ok()) << second.status().message();
  pipelines::StreamedAtlas dst = std::move(first).value();
  pipelines::StreamedAtlas src = std::move(second).value();

  Frame* one = upload_and_draw(dst, 1, solid(kRed));
  ASSERT_NE(one, nullptr);
  Frame* two = upload_and_draw(src, 2, solid(kGreen));
  ASSERT_NE(two, nullptr);
  const VkImage adopted = src.picture()->handle();

  dst = std::move(src);
  EXPECT_EQ(dst.slot_count(), 3u);
  ASSERT_TRUE(dst.has_picture());
  EXPECT_EQ(dst.picture()->handle(), adopted);
  EXPECT_FALSE(src.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_GE(timeline_->value().value(), 1u) << "dst's frame 1 was waited for";

  Frame* three = upload_and_draw(dst, 3, solid(kBlue));
  ASSERT_NE(three, nullptr);
  EXPECT_EQ(drawn(*one), solid(kRed));
  EXPECT_EQ(drawn(*two), solid(kGreen));
  EXPECT_EQ(drawn(*three), solid(kBlue));
}

TEST_F(StreamedAtlasTest, SelfMoveAssignKeepsTheAtlas) {
  auto made = make_atlas(2);
  ASSERT_TRUE(made.ok()) << made.status().message();
  pipelines::StreamedAtlas atlas = std::move(made).value();
  Frame* one = upload_and_draw(atlas, 1, solid(kRed));
  ASSERT_NE(one, nullptr);
  const VkImage picture = atlas.picture()->handle();

  // Launder through a pointer so -Wself-move does not fire under -Werror.
  pipelines::StreamedAtlas* alias = &atlas;
  atlas = std::move(*alias);
  EXPECT_TRUE(atlas.valid());
  ASSERT_TRUE(atlas.has_picture());
  EXPECT_EQ(atlas.picture()->handle(), picture);

  Frame* two = begin_frame(2);
  ASSERT_NE(two, nullptr);
  draw(*two, atlas.use(2));
  submit(*two);
  EXPECT_EQ(drawn(*one), solid(kRed));
  EXPECT_EQ(drawn(*two), solid(kRed));
}

// record_image_update's default scope waits for earlier fragment reads of the
// image: a frame that draws the image, rewrites it and draws it again, all in
// one command buffer, draws each picture once, with no hazard reported.
TEST_F(StreamedAtlasTest, ImageUpdateWaitsForEarlierReadsInTheFrame) {
  const Picture red = solid(kRed);
  vg::ImageUploadDesc upload;
  upload.extent = {kSide, kSide};
  upload.format = kFormat;
  upload.pixels = red.data();
  upload.size = sizeof(red);
  auto image = vg::upload_texture(*device_, *allocator_, upload);
  ASSERT_TRUE(image.ok()) << image.status().message();
  auto sampler = vg::Sampler::create(device());
  ASSERT_TRUE(sampler.ok()) << sampler.status().message();
  const VkDescriptorPoolSize pool_size{
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
  auto pool = vkc::DescriptorPool::create(device(), &pool_size, 1, 1);
  ASSERT_TRUE(pool.ok()) << pool.status().message();
  auto set = pool.value().allocate(pipeline_->descriptor_set_layout(0));
  ASSERT_TRUE(set.ok()) << set.status().message();
  set.value().write_combined_image_sampler(
      0, image.value().view(), sampler.value().handle(),
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  const Picture green = solid(kGreen);
  vkc::BufferDesc desc;
  desc.size = sizeof(green);
  desc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  desc.memory = vkc::MemoryUsage::Staging;
  auto source = allocator_->create_buffer(desc);
  ASSERT_TRUE(source.ok()) << source.status().message();
  std::memcpy(source.value().mapped(), green.data(), sizeof(green));
  VkBufferImageCopy whole{};
  whole.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  whole.imageExtent = {kSide, kSide, 1};

  // Frame 1 draws the image into its target, rewrites it, then draws it
  // again into a second target.
  Frame* frame = begin_frame(1);
  ASSERT_NE(frame, nullptr);
  Frame again;
  again.number = 1;
  auto target = make_target();
  ASSERT_TRUE(target.ok()) << target.status().message();
  again.target = std::move(target).value();

  draw(*frame, set.value().handle());
  const vkc::Status updated = vg::record_image_update(
      frame->cmd.handle(), source.value(), image.value(), &whole, 1);
  EXPECT_TRUE(updated.ok()) << updated.message();
  draw(frame->cmd.handle(), again.target, set.value().handle());
  submit(*frame);
  EXPECT_EQ(drawn(*frame), red);
  EXPECT_EQ(drawn(again), green);
}

}  // namespace
