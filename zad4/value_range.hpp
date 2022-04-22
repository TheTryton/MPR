#pragma once

#include <assert.h>

template<typename T>
struct value_range
{
    T low;
    T high;

    [[nodiscard]] T length() const noexcept { return high - low; }
    [[nodiscard]] bool contains(const T& v) const noexcept { return v >= low && v < high; }
    [[nodiscard]] value_range<T> split_index(size_t count, size_t index) const noexcept
    {
        assert(index < count);
        auto base = low;
        auto split_length = length() / count;
        auto split_low = base + index * split_length;
        auto split_high = split_low + split_length;
        return {
            .low = split_low,
            .high = split_high,
        };
    }
};
