// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file core/impl/retire_list.hpp
/// Header-bound implementation of the `RetireList<Key>` template. Included from
/// `core/retire_list.hpp`; not a standalone header.

#include <utility>

namespace volumetric_kit::gfx {

template <class Key>
void RetireList<Key>::push(Key key, std::function<void()> deleter) {
  entries_.push_back(Entry{std::move(key), std::move(deleter)});
}

template <class Key>
template <class ReadyFn>
std::size_t RetireList<Key>::poll(ReadyFn is_ready) {
  // Decide readiness and compact the survivors *before* running any deleter:
  // a deleter that throws (or re-enters push()) then cannot leave an
  // already-run entry behind for run_all()/the next poll to run a second time.
  //
  // Reuse ready_scratch_'s capacity, but hold it in a local for the duration so
  // a deleter that re-enters poll() operates on a fresh buffer and cannot
  // corrupt the one this call is iterating.
  std::vector<std::function<void()>> ready = std::move(ready_scratch_);
  ready.clear();
  // Allocate before moving any closures or compacting the pending entries.
  // An allocation failure then leaves every pending owner intact for retry.
  ready.reserve(entries_.size());
  std::size_t keep = 0;
  for (std::size_t i = 0; i < entries_.size(); ++i) {
    if (is_ready(entries_[i].key)) {
      ready.push_back(std::move(entries_[i].deleter));
    } else {
      // Stable-compact the not-yet-ready entries toward the front.
      if (keep != i) {
        entries_[keep] = std::move(entries_[i]);
      }
      ++keep;
    }
  }
  entries_.resize(keep);
  run_each(ready);
  const std::size_t ran = ready.size();
  ready.clear();
  ready_scratch_ = std::move(ready);  // return the buffer for the next poll()
  return ran;
}

template <class Key>
template <class WaitFn>
void RetireList<Key>::drain(WaitFn wait) {
  drain_until([&wait](const Key& key) {
    wait(key);
    return true;
  });
}

template <class Key>
template <class WaitFn>
bool RetireList<Key>::drain_until(WaitFn wait) {
  std::size_t released = 0;
  const auto compact = [&] {
    entries_.erase(entries_.begin(), entries_.begin() + released);
  };
  try {
    while (released < entries_.size()) {
      if (!wait(entries_[released].key)) {
        compact();
        return false;
      }
      // Moving out clears the completed owner before user code runs. Indices
      // survive a reentrant push() and no references span that call. Compact
      // the prefix once, including when a wait throws, so draining is linear.
      std::function<void()> deleter = std::move(entries_[released].deleter);
      ++released;
      if (deleter) deleter();
    }
  } catch (...) {
    compact();
    throw;
  }
  compact();
  return true;
}

template <class Key>
void RetireList<Key>::run_all() noexcept {
  // Loop until empty so a deleter that re-enters push() is still run (see
  // drain()).
  while (!entries_.empty()) {
    std::vector<Entry> pending = std::move(entries_);
    entries_.clear();
    for (Entry& entry : pending) {
      if (entry.deleter) {
        entry.deleter();
      }
    }
  }
}

template <class Key>
void RetireList<Key>::run_each(std::vector<std::function<void()>>& deleters) {
  for (std::function<void()>& deleter : deleters) {
    if (deleter) {
      deleter();
    }
  }
}

}  // namespace volumetric_kit::gfx
