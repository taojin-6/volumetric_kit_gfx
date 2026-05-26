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
  std::vector<std::function<void()>> ready;
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
  return ready.size();
}

template <class Key>
template <class WaitFn>
void RetireList<Key>::drain(WaitFn wait) {
  // Detach the entries first so a throwing deleter can't leave run entries to
  // be re-run at destruction; unrun deleters still release via their captures.
  std::vector<Entry> pending = std::move(entries_);
  entries_.clear();
  for (Entry& entry : pending) {
    wait(entry.key);
    if (entry.deleter) {
      entry.deleter();
    }
  }
}

template <class Key>
void RetireList<Key>::run_all() noexcept {
  std::vector<Entry> pending = std::move(entries_);
  entries_.clear();
  for (Entry& entry : pending) {
    if (entry.deleter) {
      entry.deleter();
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
