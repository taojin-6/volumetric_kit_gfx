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
  // Process in rounds until nothing remains: a deleter may re-enter push() to
  // enqueue a follow-up resource, and drain() promises to run *every* deleter
  // (matching poll(), which re-checks each iteration). Detaching each round's
  // entries before running them keeps a throwing deleter from leaving a run
  // entry to be re-run at destruction; unrun deleters still release via their
  // captures.
  while (!entries_.empty()) {
    std::vector<Entry> pending = std::move(entries_);
    entries_.clear();
    for (Entry& entry : pending) {
      wait(entry.key);
      if (entry.deleter) {
        entry.deleter();
      }
    }
  }
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
