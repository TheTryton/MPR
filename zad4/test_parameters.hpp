#pragma once

#include <schedule.hpp>
#include <value_range.hpp>
#include <array>

constexpr std::array<schedule_t, 29> schedule_types
{
    static_schedule{},
    static_schedule{.chunk_size = 1},
    static_schedule{.chunk_size = 2},
    static_schedule{.chunk_size = 4},
    static_schedule{.chunk_size = 8},
    static_schedule{.chunk_size = 16},
    static_schedule{.chunk_size = 32},
    static_schedule{.chunk_size = 64},
    static_schedule{.chunk_size = 128},
    dynamic_schedule{},
    dynamic_schedule{.chunk_size = 1},
    dynamic_schedule{.chunk_size = 2},
    dynamic_schedule{.chunk_size = 4},
    dynamic_schedule{.chunk_size = 8},
    dynamic_schedule{.chunk_size = 16},
    dynamic_schedule{.chunk_size = 32},
    dynamic_schedule{.chunk_size = 64},
    dynamic_schedule{.chunk_size = 128},
    guided_schedule{},
    guided_schedule{.chunk_size = 1},
    guided_schedule{.chunk_size = 2},
    guided_schedule{.chunk_size = 4},
    guided_schedule{.chunk_size = 8},
    guided_schedule{.chunk_size = 16},
    guided_schedule{.chunk_size = 32},
    guided_schedule{.chunk_size = 64},
    guided_schedule{.chunk_size = 128},
    auto_schedule{},
    runtime_schedule{},
};

constexpr std::array<size_t, 14> problem_sizes
{
    1ull << 7,
    1ull << 8,
    1ull << 9,
    1ull << 10,
    1ull << 11,
    1ull << 12,
    1ull << 13,
    1ull << 14,
    1ull << 15,
    1ull << 16,
    1ull << 18,
    1ull << 20,
    1ull << 22,
    1ull << 24,
};

constexpr std::array<size_t, 10> thread_counts
{
    1,
    2,
    4,
    6,
    8,
    12,
    16,
    32,
    64,
    128,
};

constexpr std::array<value_range<float>, 7> value_ranges
{
    value_range<float>{0.0f, 1.0f},
    value_range<float>{0.0f, 10.0f},
    value_range<float>{0.0f, 100.0f},
    value_range<float>{0.0f, 1000.0f},
    value_range<float>{1.0f, 10.0f},
    value_range<float>{10.0f, 100.0f},
    value_range<float>{100.0f, 1000.0f},
};

constexpr std::array<size_t, 5> bucket_counts
{
    4,
    8,
    16,
    32,
    64,
};

constexpr std::array<double, 6> bucket_percentage_sizes
{
        1.5,
        2.5,
        2.5,
        3.0,
        3.5,
        4.0,
};

constexpr std::array<size_t, 7> single_thread_sort_thresholds
{
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
};

struct insertion_sort_tag{};
struct bubble_sort_tag{};
struct selection_sort_tag{};
struct std_sort_tag{};
struct std_stable_sort_tag{};

using n2_sort_t = std::variant<
    insertion_sort_tag,
    bubble_sort_tag,
    selection_sort_tag,
    std_sort_tag,
    std_stable_sort_tag
>;

constexpr std::array<n2_sort_t, 5> n2_sorts
{
    insertion_sort_tag{},
    bubble_sort_tag{},
    selection_sort_tag{},
    std_sort_tag{},
    std_stable_sort_tag{},
};
