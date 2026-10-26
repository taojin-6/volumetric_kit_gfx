// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <new>

#include "allocation_failure.hpp"

TEST(AllocationFailure, NestedExclusionsPreservePendingFailure) {
  bool excluded_threw = false;
  bool caller_threw = false;
  vg_test::AllocationFailure failure;
  try {
    const vg_test::AllocationExclusion outer;
    {
      const vg_test::AllocationExclusion inner;
      ::operator delete(::operator new(17));
    }
    ::operator delete(::operator new(17));
  } catch (const std::bad_alloc&) {
    excluded_threw = true;
  }
  try {
    ::operator delete(::operator new(17));
  } catch (const std::bad_alloc&) {
    caller_threw = true;
  }
  const bool triggered = failure.stop();
  EXPECT_FALSE(excluded_threw);
  EXPECT_TRUE(caller_threw);
  EXPECT_TRUE(triggered);
  EXPECT_EQ(failure.allocations(), 1u);
}

TEST(AllocationFailure, ExcludedAllocationsAreNotRecorded) {
  vg_test::record_allocations(17);
  {
    const vg_test::AllocationExclusion exclusion;
    ::operator delete(::operator new(17));
  }
  void* caller = ::operator new(17);
  const void* site = vg_test::sole_allocation_site();
  ::operator delete(caller);
  EXPECT_NE(site, nullptr);
}
