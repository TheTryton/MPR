#pragma once

#include <buckets.hpp>
#include <algorithm>

template<typename T>
struct default_key
{
    T operator()(const T& element) { return element; }
};

template<typename It, typename KeyF = default_key<typename std::iterator_traits<It>::value_type>>
void bucket_sort(
    It first, It last,
    const range<typename std::iterator_traits<It>::value_type>& value_range,
    KeyF keyf = {},
    size_t bucket_count = 32)
{
    using T = typename std::iterator_traits<It>::value_type;
    using BT = variable_size_bucket_t<T, std::allocator<T>>;

    auto length = std::distance(first, last);
    auto bucket_size = length * 2 / bucket_count;
    buckets_t<T, std::allocator<T>, BT, std::allocator<BT>> buckets{bucket_count, bucket_size, value_range};

    std::for_each(first, last, [&buckets](T& v)
    {
        buckets.insert(std::move(v));
    });

    for(auto& bucket : buckets)
    {
        std::stable_sort(std::begin(bucket), std::end(bucket), [&keyf](auto&& a, auto&& b)
        {
            return keyf(a) < keyf(b);
        });
    }

    for(auto& bucket : buckets)
    {
        first = std::move(std::begin(bucket), std::end(bucket), first);
    }
}
