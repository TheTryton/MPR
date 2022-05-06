#pragma once

#include <assert.h>
#include <concepts>

template<typename KeyType>
concept RangeBoundary =
#ifdef _MSC_VER
true;
#else
std::is_copy_constructible_v<KeyType> && requires(KeyType key, size_t size)
{
    { key >= key } -> std::convertible_to<bool>;
    { key < key } -> std::convertible_to<bool>;
    { key - key } -> std::convertible_to<KeyType>;
    { key + key } -> std::convertible_to<KeyType>;
    { key / size } -> std::convertible_to<KeyType>;
    { key * size } -> std::convertible_to<KeyType>;
};
#endif

template<RangeBoundary KeyType, typename SizeType = size_t>
struct value_range
{
    using value_type = KeyType;
    using reference = value_type&;
    using const_reference = const value_type&;

    value_type low;
    value_type high;

    [[nodiscard]] value_type length() const noexcept
    {
        return high - low;
    }
    [[nodiscard]] bool is_reverted() const noexcept
    {
        return high < low;
    }
    [[nodiscard]] bool contains(const_reference v) const noexcept
    { 
        return v >= low && v < high;
    }
    [[nodiscard]] value_range<value_type> split_index(size_t index, size_t count) const noexcept
    {
        assert(index < count);
        auto base = low;
        auto split_length = length() / count;
        auto split_low = base + split_length * index;
        auto split_high = split_low + split_length;
        return {
            .low = split_low,
            .high = split_high,
        };
    }
    template<typename KeyFunc>
    [[nodiscard]] value_range<std::invoke_result_t<KeyFunc, const value_type&>> transform(KeyFunc key_func) const
    {
        auto low_f = key_func(low);
        auto high_f = key_func(high);

        return {
            .low = std::min(low_f, high_f),
            .high = std::max(low_f, high_f)
        };
    }
};
