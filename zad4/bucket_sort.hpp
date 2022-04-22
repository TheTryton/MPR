#pragma once

#include <buckets.hpp>
#include <algorithm>
#include <execution>

template<typename T>
struct default_key
{
    T operator()(const T& element) { return element; }
};

template<typename ExecutionPolicy, typename It, typename KeyF, typename BT>
void bucket_sort_impl(
    ExecutionPolicy execution_policy,
    It first, It last,
    const value_range<typename std::iterator_traits<It>::value_type>& value_range,
    KeyF keyf = {},
    size_t bucket_count = 32,
    double base_bucket_size_multiplier = 2.0)
{
    using T = typename std::iterator_traits<It>::value_type;

    auto length = std::distance(first, last);
    auto bucket_size = static_cast<size_t>(length * base_bucket_size_multiplier / bucket_count);
    thread_safe_buckets_t<T, std::allocator<T>, BT> buckets{bucket_count, bucket_size, value_range};

    std::for_each(execution_policy, first, last, [&buckets](T& v)
    {
        buckets.insert(std::move(v));
    });

    for(auto& bucket : buckets)
    {
        std::stable_sort(execution_policy, std::begin(bucket), std::end(bucket), [&keyf](auto&& l, auto&& r)
        {
            return keyf(l) < keyf(r);
        });
    }

    for(auto& bucket : buckets)
    {
        first = std::move(execution_policy, std::begin(bucket), std::end(bucket), first);
    }
}

template<typename It, typename KeyF, typename BT>
constexpr void bucket_sort_impl(
    std::execution::sequenced_policy,
    It first, It last,
    const value_range<typename std::iterator_traits<It>::value_type>& value_range,
    KeyF keyf = {},
    size_t bucket_count = 32,
    double base_bucket_size_multiplier = 2.0)
{
    using T = typename std::iterator_traits<It>::value_type;

    auto length = std::distance(first, last);
    auto bucket_size = static_cast<size_t>(length * base_bucket_size_multiplier / bucket_count);
    buckets_t<T, std::allocator<T>, BT, std::allocator<BT>> buckets{bucket_count, bucket_size, value_range};

    std::for_each(first, last, [&buckets](T& v)
    {
        buckets.insert(std::move(v));
    });

    for(auto& bucket : buckets)
    {
        std::stable_sort(std::begin(bucket), std::end(bucket), [&keyf](auto&& l, auto&& r)
        {
            return keyf(l) < keyf(r);
        });
    }

    for(auto& bucket : buckets)
    {
        first = std::move(std::begin(bucket), std::end(bucket), first);
    }
}

template<
    typename It,
    typename KeyF = default_key<typename std::iterator_traits<It>::value_type>
>
constexpr void bucket_sort(
    It first, It last,
    const value_range<typename std::iterator_traits<It>::value_type>& value_range,
    KeyF keyf = {},
    size_t bucket_count = 32,
    double base_bucket_size_multiplier = 2.0,
    bool use_variable_size_bucket = true)
{
    if (use_variable_size_bucket)
    {
        bucket_sort_impl<It, KeyF,
            variable_size_bucket_t<
                typename std::iterator_traits<It>::value_type,
                std::allocator<typename std::iterator_traits<It>::value_type>
            >>
            (std::execution::seq, first, last, value_range, keyf, bucket_count, base_bucket_size_multiplier);
    }
    else
    {
        bucket_sort_impl<It, KeyF,
            fixed_size_bucket_t<
                typename std::iterator_traits<It>::value_type,
                std::allocator<typename std::iterator_traits<It>::value_type>
            >>
            (std::execution::seq, first, last, value_range, keyf, bucket_count, base_bucket_size_multiplier);
    }
}

template<
    typename ExecutionPolicy,
    typename It,
    typename KeyF = default_key<typename std::iterator_traits<It>::value_type>>
void bucket_sort(
    ExecutionPolicy execution_policy,
    It first, It last,
    const value_range<typename std::iterator_traits<It>::value_type>& value_range,
    KeyF keyf = {},
    size_t bucket_count = 32,
    double base_bucket_size_multiplier = 2.0,
    bool use_variable_size_bucket = true)
{
    if (use_variable_size_bucket)
    {
        bucket_sort_impl<ExecutionPolicy, It, KeyF,
            variable_size_bucket_t<
                typename std::iterator_traits<It>::value_type,
                std::allocator<typename std::iterator_traits<It>::value_type>
            >>
            (execution_policy, first, last, value_range, keyf, bucket_count, base_bucket_size_multiplier);
    }
    else
    {
        bucket_sort_impl<ExecutionPolicy, It, KeyF,
            fixed_size_bucket_t<
                typename std::iterator_traits<It>::value_type,
                std::allocator<typename std::iterator_traits<It>::value_type>
            >>
            (execution_policy, first, last, value_range, keyf, bucket_count, base_bucket_size_multiplier);
    }
}