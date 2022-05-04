#pragma once

#include <buckets.hpp>
#include <algorithm>
#include <execution>
#include <parallel_primitives.hpp>
#include <memory_resource>
#include <iostream>
#include <syncstream>

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
        std::stable_sort(std::begin(bucket), std::end(bucket), [&keyf](auto&& l, auto&& r)
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

template<typename T>
concept Key = std::floating_point<T> || std::integral<T>;


template<typename UnaryOp, typename It>
concept KeyUnaryOp = requires (It it, UnaryOp unary_op)
{
    { unary_op(*it) } -> Key;
};

static std::atomic<size_t> counting_allocator_allocations = 0;
static std::atomic<size_t> counting_allocator_memory = 0;

template<typename T>
class counting_allocator
{
private:
    std::allocator<T> _allocator;
public:
    using value_type = typename std::allocator<T>::value_type;
    using size_type = typename std::allocator<T>::size_type;
    using difference_type = typename std::allocator<T>::difference_type;
    using propagate_on_container_move_assignment = typename std::allocator<T>::propagate_on_container_move_assignment;
public:
    constexpr counting_allocator() noexcept = default;
    constexpr counting_allocator(const counting_allocator&) noexcept = default;
    constexpr counting_allocator(counting_allocator&&) noexcept = default;
public:
    constexpr ~counting_allocator()
    {
        std::osyncstream(std::cout) << "[allocations, bytes]: " << counting_allocator_allocations << ", " << counting_allocator_memory/1e9 << "GB" << std::endl;
    }
public:
    [[nodiscard]] constexpr value_type* allocate(size_type n)
    {
        ++counting_allocator_allocations;
        counting_allocator_memory += n;
        return _allocator.allocate(n);
    }
    constexpr void deallocate(value_type* p, size_type n)
    {
        _allocator.deallocate(p, n);
    }
public:
    constexpr bool operator==(const counting_allocator& other) const noexcept { return _allocator == other._allocator; }
    constexpr bool operator!=(const counting_allocator& other) const noexcept { return _allocator != other._allocator; }
};

constexpr size_t desired_bucket_size_coeff = 3;

template<std::random_access_iterator IORandIt>
void bucket_sort_v1(
    IORandIt first, IORandIt last,
    const value_range<typename std::iterator_traits<IORandIt>::value_type>& value_range,
    size_t desired_bucket_size = 32)
{
    using T = typename std::iterator_traits<IORandIt>::value_type;
    using TAlloc = std::allocator<T>;
    using BT = variable_size_bucket_t<T, TAlloc>;
    using BTAlloc = std::allocator<BT>;

    fixed_size_dynamic_array<size_t> thread_buckets_sizes{};
    fixed_size_dynamic_array<size_t> thread_buckets_offsets{};

    #pragma omp parallel default(none) shared(thread_buckets_sizes, thread_buckets_offsets, value_range) firstprivate(first, last, desired_bucket_size)
    {
        // prepare memory for offset table for each thread
        #pragma omp single
        {
            thread_buckets_sizes = fixed_size_dynamic_array<size_t>(parallel::total_thread_count(), static_cast<size_t>(0));
            thread_buckets_offsets = fixed_size_dynamic_array<size_t>(parallel::total_thread_count(), static_cast<size_t>(0));
        }

        // extract current thread info
        auto total_thread_count = parallel::total_thread_count();
        auto current_thread_index = parallel::current_thread_index();
        auto current_thread_value_range = value_range.split_index(current_thread_index, total_thread_count);
        auto estimate_elements_for_thread = (double)std::distance(first, last) / total_thread_count;
        auto buckets_count = static_cast<size_t>(std::ceil(estimate_elements_for_thread / desired_bucket_size));
        // create buckets
        buckets_t<T, TAlloc, BT, BTAlloc> buckets{ buckets_count, desired_bucket_size * desired_bucket_size_coeff, current_thread_value_range };

        // read whole array and insert values into buckets if they fall into current thread value range
        std::for_each(first, last, [&buckets, &current_thread_value_range](auto&& v)
            {
                if (current_thread_value_range.contains(v))
                {
                    buckets.insert(v);
                }
            });

        // sort each bucket with regular sequential sorting algorithm
        for (auto& bucket : buckets)
        {
            std::sort(std::begin(bucket), std::end(bucket));
        }

        // calculate total amount of elements inserted into current thread's
        // buckets and place it in thread_buckets_sizes
        thread_buckets_sizes[current_thread_index] = buckets.total_size();

        // wait for all threads to finish doing above operations and calculate final insert offsets to original array
        #pragma omp barrier
        #pragma omp single
        {
            std::partial_sum(std::begin(thread_buckets_sizes), std::end(thread_buckets_sizes) - 1, std::begin(thread_buckets_offsets) + 1);
        }

        // get insert offset for current thread
        auto thread_insert_offset = thread_buckets_offsets[current_thread_index];
        auto d_it = first + thread_insert_offset;
        //std::advance(d_it, thread_insert_offset);

        // insert sorted buckets into destination range starting at calculated offset
        for (auto& bucket : buckets)
        {
            for (auto& element : bucket)
            {
                *d_it++ = element;
            }
        }
    }
}

template<std::random_access_iterator IORandIt>
void bucket_sort_v2(
    IORandIt first, IORandIt last,
    const value_range<typename std::iterator_traits<IORandIt>::value_type>& value_range,
    size_t desired_bucket_size = 32)
{
    using T = typename std::iterator_traits<IORandIt>::value_type;
    using TAlloc = std::allocator<T>;
    using BT = threadsafe::lockfree_fixed_size_bucket_t<T, TAlloc, fixed_size_dynamic_array<T, TAlloc>>;
    using BTAlloc = std::allocator<BT>;

    // calculate total length and buckets count based on desired bucket size
    auto total_length = std::distance(first, last);
    auto buckets_count = static_cast<size_t>(std::ceil((double)total_length / desired_bucket_size));

    // create thread safe buckets
    buckets_t<T, TAlloc, BT, BTAlloc> buckets{ buckets_count, desired_bucket_size * desired_bucket_size_coeff, value_range };

    // distribute inserting across all threads
    #pragma omp parallel for default(none) firstprivate(first, total_length) shared(buckets)
    for (parallel::index_t offset = 0; offset < total_length; ++offset)
    {
        buckets.insert(first[offset]);
    }

    // prepare memory for offset table for each bucket
    fixed_size_dynamic_array<size_t> bucket_sizes(buckets.size(), static_cast<size_t>(0));
    fixed_size_dynamic_array<size_t> bucket_offsets(buckets.size(), static_cast<size_t>(0));

    // sort buckets across all threads and insert their sizes into bucket_sizes
    auto buckets_first = std::begin(buckets);
    auto bucket_sizes_first = std::begin(bucket_sizes);
    auto buckets_last = std::end(buckets);
    auto total_buckets_length = std::distance(buckets_first, buckets_last);

    #pragma omp parallel for default(none) firstprivate(buckets_first, bucket_sizes_first, total_buckets_length)
    for (parallel::index_t offset = 0; offset < total_buckets_length; ++offset)
    {
        std::sort(std::begin(buckets_first[offset]), std::end(buckets_first[offset]));
        bucket_sizes_first[offset] = buckets_first[offset].size();
    }

    // calculate bucket insert offsets
    // TODO: parallelize?
    std::partial_sum(std::begin(bucket_sizes), std::end(bucket_sizes) - 1, std::begin(bucket_offsets) + 1);

    // insert sorted buckets into destination range starting at calculated offset
    auto bucket_offsets_first = std::begin(bucket_offsets);

    #pragma omp parallel for default(none) firstprivate(buckets_first, bucket_offsets_first, first, total_buckets_length)
    for (parallel::index_t offset = 0; offset < total_buckets_length; ++offset)
    {
        auto d_it = first + bucket_offsets_first[offset];
        for (auto& e : buckets_first[offset])
        {
            *d_it++ = e;
        }
    }
}

template<
    std::random_access_iterator IORandIt
>
void bucket_sort_v3(
    IORandIt first, IORandIt last,
    const value_range<typename std::iterator_traits<IORandIt>::value_type>& value_range,
    size_t desired_bucket_size = 32)
{
    using T = typename std::iterator_traits<IORandIt>::value_type;
    using TAlloc = std::allocator<T>; //std::pmr::polymorphic_allocator<T>;
    using BT = variable_size_bucket_t<T, TAlloc>;
    using BTAlloc = std::allocator<BT>; //std::pmr::polymorphic_allocator<BT>;
    //using STAlloc = std::pmr::polymorphic_allocator<size_t>;

    auto total_length = std::distance(first, last);
    auto buckets_count = static_cast<size_t>(std::ceil((double)total_length / desired_bucket_size));

    //auto buffer_resource = std::pmr::monotonic_buffer_resource(total_length, std::pmr::get_default_resource());
    //auto memory_resource = std::pmr::synchronized_pool_resource(&buffer_resource);
    //auto t_allocator = TAlloc(&memory_resource);
    //auto bt_allocator = BTAlloc(&memory_resource);
    //auto st_allocator = STAlloc(&memory_resource);

    // prepare memory for offset table for each bucket
    fixed_size_dynamic_array<size_t> bucket_sizes(buckets_count, static_cast<size_t>(0));
    fixed_size_dynamic_array<size_t> bucket_offsets(buckets_count, static_cast<size_t>(0));
    fixed_size_dynamic_array<size_t> bucket_advancing_offsets(buckets_count, static_cast<size_t>(0));

    #pragma omp parallel default(none) shared(bucket_sizes, bucket_offsets, bucket_advancing_offsets, value_range) firstprivate(first, buckets_count, desired_bucket_size, total_length)
    {
        // create local buckets
        auto adjusted_desired_bucket_size = static_cast<size_t>(ceil(desired_bucket_size / (double)parallel::total_thread_count() * desired_bucket_size_coeff));
        buckets_t<T, TAlloc, BT, BTAlloc> buckets{ buckets_count, adjusted_desired_bucket_size, value_range};

        // read array part and insert values into local thread buckets
        #pragma omp for
        for (parallel::index_t offset = 0; offset < total_length; ++offset)
        {
            buckets.insert(first[offset]);
        }

        // add each bucket size to final total bucket size atomically
        {
            auto buckets_first = std::begin(buckets);
            auto bucket_sizes_first = std::begin(bucket_sizes);
            auto buckets_last = std::end(buckets);
            for (; buckets_first != buckets_last; ++buckets_first, ++bucket_sizes_first)
            {
                std::atomic_ref ref_to_bucket_size{ *bucket_sizes_first };
                ref_to_bucket_size.fetch_add(buckets_first->size());
            }
        }

        // wait for all threads to finish above steps
        // and calculate offsets of each final
        #pragma omp barrier
        #pragma omp single
        {
            std::partial_sum(std::begin(bucket_sizes), std::end(bucket_sizes) - 1, std::begin(bucket_offsets) + 1);
        }

        // copy offsets to advancing offsets
        auto offsets_first = std::begin(bucket_offsets);
        auto advancing_offsets_first = std::begin(bucket_advancing_offsets);
        #pragma omp for
        for (parallel::index_t offset = 0; offset < buckets_count; ++offset)
        {
            advancing_offsets_first[offset] = offsets_first[offset];
        }

        // concatenate buckets "inplace" (insert them back to original range using advancing offsets atomically updating them)
        {
            auto buckets_first = std::begin(buckets);
            auto buckets_last = std::end(buckets);
            auto advancing_offsets_first = std::begin(bucket_advancing_offsets);
            for (; buckets_first != buckets_last; ++buckets_first, ++advancing_offsets_first)
            {
                std::atomic_ref ref_to_bucket_offset{ *advancing_offsets_first };
                auto local_offset = ref_to_bucket_offset.fetch_add(buckets_first->size());
                std::copy(std::begin(*buckets_first), std::end(*buckets_first), first + local_offset);
            }
        }
    }

    // finally sort concatenated buckets across all threads
    auto bucket_offsets_first = std::begin(bucket_offsets);
    auto bucket_sizes_first = std::begin(bucket_sizes);
    #pragma omp parallel for default(none) firstprivate(first, buckets_count, bucket_offsets_first, bucket_sizes_first)
    for (parallel::index_t offset = 0; offset < buckets_count; ++offset)
    {
        auto bucket_first = first + bucket_offsets_first[offset];
        auto bucket_last = bucket_first + bucket_sizes_first[offset];
        std::sort(bucket_first, bucket_last);
    }
}