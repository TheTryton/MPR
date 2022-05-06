#pragma once

#include <buckets.hpp>
#include <algorithm>
#include <execution>
#include <parallel_primitives.hpp>
#include <memory_resource>
#include <iostream>
#include <syncstream>

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

constexpr double default_bucket_size_coeff = 3;

struct bucket_sort_measurement
{
    millis_double bucketization;
    millis_double sequential_sorting;
    millis_double writing_back;
    std::optional<millis_double> concatenation;
    millis_double total;
};

struct bucket_sort_stats
{
    mean_stddev bucketization;
    mean_stddev sequential_sorting;
    mean_stddev writing_back;
    mean_stddev concatenation;
    mean_stddev total;
};

bucket_sort_measurement operator+(const bucket_sort_measurement& l, const bucket_sort_measurement& r)
{
    return bucket_sort_measurement{
        .bucketization = l.bucketization + r.bucketization,
        .sequential_sorting = l.sequential_sorting + r.sequential_sorting,
        .writing_back = l.writing_back + r.writing_back,
        .concatenation = !l.concatenation && !r.concatenation ? std::optional<millis_double>() : 
        (
            (l.concatenation ? *l.concatenation : millis_double{}) + (r.concatenation ? *r.concatenation : millis_double{})
        ),
        .total = l.total + r.total,
    };
}

namespace detail
{
    template<
        typename BucketType,
        typename IORandIt,
        typename KeyFunc,
        typename ElementAllocator,
        typename OffsetsAllocator,
        typename BucketAllocator
    >
    bucket_sort_measurement bucket_sort_v1_parallel_region_0(
        // shared
        fixed_size_dynamic_array<size_t, OffsetsAllocator>& thread_buckets_sizes,
        fixed_size_dynamic_array<size_t, OffsetsAllocator>& thread_buckets_offsets,
        const value_range<std::invoke_result_t<KeyFunc, const typename std::iterator_traits<IORandIt>::value_type&>>& value_range,
        KeyFunc key_func,
        ElementAllocator& element_allocator,
        OffsetsAllocator& offsets_allocator,
        BucketAllocator& bucket_allocator,
        // firstprivate
        IORandIt first,
        IORandIt last,
        size_t desired_bucket_size,
        double final_bucket_size_coeff
    )
    {
        using ElementType = typename std::iterator_traits<IORandIt>::value_type;

        // create stats object

        bucket_sort_measurement local_stats;

        // extract current thread info
        auto total_thread_count = parallel::total_thread_count();
        auto current_thread_index = parallel::current_thread_index();
        auto current_thread_value_range = value_range.split_index(current_thread_index, total_thread_count);
        auto estimate_elements_for_thread = (double)std::distance(first, last) / total_thread_count;
        auto buckets_count = static_cast<size_t>(std::ceil(estimate_elements_for_thread / desired_bucket_size));

        // BUCKETIZATION

        auto start = std::chrono::high_resolution_clock::now();

        // create buckets
        buckets_t<ElementType, KeyFunc, ElementAllocator, BucketType, BucketAllocator> buckets{
            buckets_count,
            static_cast<size_t>(std::ceil(desired_bucket_size * final_bucket_size_coeff)),
            current_thread_value_range,
            key_func,
            element_allocator,
            bucket_allocator
        };

        // read whole array and insert values into buckets if they fall into current thread value range
        std::for_each(first, last, [&buckets, &current_thread_value_range, &key_func](auto&& v)
            {
                if (current_thread_value_range.contains(key_func(v)))
                {
                    buckets.insert(v);
                }
            });

        auto end = std::chrono::high_resolution_clock::now();

        local_stats.bucketization = std::chrono::duration_cast<millis_double>(end - start);

        // SEQUENTIAL_SORTING

        start = std::chrono::high_resolution_clock::now();

        // sort each bucket with regular sequential sorting algorithm
        for (auto& bucket : buckets)
        {
            std::stable_sort(std::begin(bucket), std::end(bucket), [&key_func](auto&& l, auto&& r){ return key_func(l) < key_func(r); });
        }

        end = std::chrono::high_resolution_clock::now();

        local_stats.sequential_sorting = std::chrono::duration_cast<millis_double>(end - start);
        
        // WRITING_BACK

        start = std::chrono::high_resolution_clock::now();

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

        end = std::chrono::high_resolution_clock::now();

        local_stats.writing_back = std::chrono::duration_cast<millis_double>(end - start);

        return local_stats;
    }
}

template<
    std::random_access_iterator IORandIt,
    typename KeyFunc = identity_key<typename std::iterator_traits<IORandIt>::value_type>,
    Allocator<typename std::iterator_traits<IORandIt>::value_type> ElementAllocator = std::allocator<typename std::iterator_traits<IORandIt>::value_type>,
    Allocator<size_t> OffsetsAllocator = std::allocator<size_t>,
    Bucket<typename std::iterator_traits<IORandIt>::value_type, ElementAllocator> BucketType = variable_size_bucket_t<typename std::iterator_traits<IORandIt>::value_type, ElementAllocator>,
    Allocator<BucketType> BucketAllocator = std::allocator<BucketType>
>
requires BucketsKey<std::invoke_result_t<KeyFunc, const typename std::iterator_traits<IORandIt>::value_type&>>
bucket_sort_measurement bucket_sort_v1(
    IORandIt first, IORandIt last,
    const value_range<std::invoke_result_t<KeyFunc, const typename std::iterator_traits<IORandIt>::value_type&>>& value_range,
    KeyFunc key_func = {},
    size_t desired_bucket_size = 16384,
    double final_bucket_size_coeff = default_bucket_size_coeff,
    ElementAllocator element_allocator = {},
    OffsetsAllocator offsets_allocator = {},
    BucketAllocator bucket_allocator = {},
    std::optional<size_t> num_threads = std::nullopt
)
{
    bucket_sort_measurement total_stats{};

    auto start = std::chrono::high_resolution_clock::now();

    fixed_size_dynamic_array<size_t, OffsetsAllocator> thread_buckets_sizes{ *num_threads, static_cast<size_t>(0), offsets_allocator };
    fixed_size_dynamic_array<size_t, OffsetsAllocator> thread_buckets_offsets{ *num_threads, static_cast<size_t>(0), offsets_allocator };

    if (num_threads)
    {
        #pragma omp parallel default(none)\
        shared(thread_buckets_sizes, thread_buckets_offsets, value_range, key_func, element_allocator, offsets_allocator, bucket_allocator, total_stats)\
        firstprivate(first, last, desired_bucket_size, final_bucket_size_coeff)\
        num_threads(*num_threads)
        {
            auto local_stats = detail::bucket_sort_v1_parallel_region_0<BucketType>(
                thread_buckets_sizes, thread_buckets_offsets, value_range, key_func, element_allocator, offsets_allocator, bucket_allocator,
                first, last, desired_bucket_size, final_bucket_size_coeff
            );
            #pragma omp critical
            total_stats = total_stats + local_stats;
        }
    }
    else
    {
        #pragma omp parallel default(none)\
        shared(thread_buckets_sizes, thread_buckets_offsets, value_range, key_func, element_allocator, offsets_allocator, bucket_allocator, total_stats)\
        firstprivate(first, last, desired_bucket_size, final_bucket_size_coeff)
        {
            auto local_stats = detail::bucket_sort_v1_parallel_region_0<BucketType>(
                thread_buckets_sizes, thread_buckets_offsets, value_range, key_func, element_allocator, offsets_allocator, bucket_allocator,
                first, last, desired_bucket_size, final_bucket_size_coeff
            );
            #pragma omp critical
            total_stats = total_stats + local_stats;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    total_stats.total = std::chrono::duration_cast<millis_double>(end - start);
    return total_stats;
}

template<
    std::random_access_iterator IORandIt,
    typename KeyFunc = identity_key<typename std::iterator_traits<IORandIt>::value_type>,
    Allocator<typename std::iterator_traits<IORandIt>::value_type> ElementAllocator = std::allocator<typename std::iterator_traits<IORandIt>::value_type>,
    Allocator<size_t> OffsetsAllocator = std::allocator<size_t>,
    Bucket<typename std::iterator_traits<IORandIt>::value_type, ElementAllocator> BucketType = threadsafe::lockfree_fixed_size_bucket_t<typename std::iterator_traits<IORandIt>::value_type, ElementAllocator>,
    Allocator<BucketType> BucketAllocator = std::allocator<BucketType>
>
bucket_sort_measurement bucket_sort_v2(
    IORandIt first, IORandIt last,
    const value_range<std::invoke_result_t<KeyFunc, const typename std::iterator_traits<IORandIt>::value_type&>>& value_range,
    KeyFunc key_func = {},
    size_t desired_bucket_size = 16384,
    double final_bucket_size_coeff = default_bucket_size_coeff,
    ElementAllocator element_allocator = {},
    OffsetsAllocator offsets_allocator = {},
    BucketAllocator bucket_allocator = {},
    const parallel::schedule_t& total_length_schedule = parallel::guided_schedule{ .chunk_size = 32 },
    const parallel::schedule_t& buckets_count_schedule = parallel::guided_schedule{ .chunk_size = 32 },
    std::optional<size_t> num_threads = std::nullopt
)
{
    bucket_sort_measurement total_stats{};

    auto tstart = std::chrono::high_resolution_clock::now();

    using ElementType = typename std::iterator_traits<IORandIt>::value_type;

    // BUCKETIZATION

    auto start = std::chrono::high_resolution_clock::now();

    // calculate total length and buckets count based on desired bucket size
    auto total_length = std::distance(first, last);
    auto buckets_count = static_cast<size_t>(std::ceil((double)total_length / desired_bucket_size));

    // create thread safe buckets
    buckets_t<ElementType, KeyFunc, ElementAllocator, BucketType, BucketAllocator> buckets{
        buckets_count,
        static_cast<size_t>(std::ceil(desired_bucket_size * final_bucket_size_coeff)),
        value_range,
        key_func,
        element_allocator,
        bucket_allocator
    };

    // distribute inserting across all threads
    if (num_threads)
    {
        #pragma omp parallel num_threads(*num_threads) shared(buckets) firstprivate(first, total_length)
        {
            parallel::for_region(
                [&buckets, first](auto offset)
                {
                    buckets.insert(first[offset]);
                },
                0, total_length, total_length_schedule
            );
        }
    }
    else
    {
        #pragma omp parallel shared(buckets) firstprivate(first, total_length)
        {
            parallel::for_region(
                [&buckets, first](auto offset)
                {
                    buckets.insert(first[offset]);
                },
                0, total_length, total_length_schedule
            );
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    total_stats.bucketization = std::chrono::duration_cast<millis_double>(end - start);

    // SEQUENTIAL_SORTING

    start = std::chrono::high_resolution_clock::now();

    // prepare memory for offset table for each bucket
    fixed_size_dynamic_array<size_t, OffsetsAllocator> bucket_sizes{ buckets.size(), static_cast<size_t>(0), offsets_allocator };
    fixed_size_dynamic_array<size_t, OffsetsAllocator> bucket_offsets{ buckets.size(), static_cast<size_t>(0), offsets_allocator };

    // sort buckets across all threads and insert their sizes into bucket_sizes
    auto buckets_first = std::begin(buckets);
    auto bucket_sizes_first = std::begin(bucket_sizes);
    auto buckets_last = std::end(buckets);
    auto total_buckets_length = std::distance(buckets_first, buckets_last);

    if (num_threads)
    {
        #pragma omp parallel num_threads(*num_threads) shared(key_func) firstprivate(buckets_first, bucket_sizes_first, total_buckets_length)
        {
            parallel::for_region(
                [&key_func, buckets_first, bucket_sizes_first](auto offset)
                {
                    std::stable_sort(std::begin(buckets_first[offset]), std::end(buckets_first[offset]), [&key_func](auto&& l, auto&& r) { return key_func(l) < key_func(r); });
                    bucket_sizes_first[offset] = buckets_first[offset].size();
                },
                0, total_buckets_length, buckets_count_schedule
            );
        }
    }
    else
    {
        #pragma omp parallel shared(key_func) firstprivate(buckets_first, bucket_sizes_first, total_buckets_length)
        {
            parallel::for_region(
                [&key_func, buckets_first, bucket_sizes_first](auto offset)
                {
                    std::stable_sort(std::begin(buckets_first[offset]), std::end(buckets_first[offset]), [&key_func](auto&& l, auto&& r) { return key_func(l) < key_func(r); });
                    bucket_sizes_first[offset] = buckets_first[offset].size();
                },
                0, total_buckets_length, buckets_count_schedule
            );
        }
    }

    end = std::chrono::high_resolution_clock::now();

    total_stats.sequential_sorting = std::chrono::duration_cast<millis_double>(end - start);

    // WRITING_BACK

    start = std::chrono::high_resolution_clock::now();

    // calculate bucket insert offsets
    std::partial_sum(std::begin(bucket_sizes), std::end(bucket_sizes) - 1, std::begin(bucket_offsets) + 1);

    // insert sorted buckets into destination range starting at calculated offset
    auto bucket_offsets_first = std::begin(bucket_offsets);

    if (num_threads)
    {
        #pragma omp parallel num_threads(*num_threads) firstprivate(buckets_first, bucket_offsets_first, first, total_buckets_length)
        {
            parallel::for_region(
                [buckets_first, bucket_offsets_first, first](auto offset)
                {
                    auto d_it = first + bucket_offsets_first[offset];
                    for (auto& e : buckets_first[offset])
                    {
                        *d_it++ = e;
                    }
                },
                0, total_buckets_length, buckets_count_schedule
            );
        }
    }
    else
    {
        #pragma omp parallel firstprivate(buckets_first, bucket_offsets_first, first, total_buckets_length)
        {
            parallel::for_region(
                [buckets_first, bucket_offsets_first, first](auto offset)
                {
                    auto d_it = first + bucket_offsets_first[offset];
                    for (auto& e : buckets_first[offset])
                    {
                        *d_it++ = e;
                    }
                },
                0, total_buckets_length, buckets_count_schedule
            );
        }
    }

    end = std::chrono::high_resolution_clock::now();

    total_stats.writing_back = std::chrono::duration_cast<millis_double>(end - start);

    auto tend = std::chrono::high_resolution_clock::now();

    total_stats.total = std::chrono::duration_cast<millis_double>(tend - tstart);
    return total_stats;
}

namespace detail
{
    template<
        typename BucketType,
        typename IORandIt,
        typename KeyFunc,
        typename ElementAllocator,
        typename OffsetsAllocator,
        typename BucketAllocator
    >
    bucket_sort_measurement bucket_sort_v3_parallel_region_0(
        // shared
        fixed_size_dynamic_array<size_t, OffsetsAllocator>& bucket_sizes,
        fixed_size_dynamic_array<size_t, OffsetsAllocator>& bucket_offsets,
        fixed_size_dynamic_array<size_t, OffsetsAllocator>& bucket_advancing_offsets,
        const value_range<std::invoke_result_t<KeyFunc, const typename std::iterator_traits<IORandIt>::value_type&>>& value_range,
        KeyFunc key_func,
        ElementAllocator& element_allocator,
        BucketAllocator& bucket_allocator,
        // firstprivate
        IORandIt first,
        size_t buckets_count,
        size_t desired_bucket_size,
        double final_bucket_size_coeff,
        size_t total_length,
        // options
        const parallel::schedule_t& total_length_schedule,
        const parallel::schedule_t& buckets_count_schedule
    )
    {
        bucket_sort_measurement local_stats;

        using ElementType = typename std::iterator_traits<IORandIt>::value_type;

        // BUCKETIZATION

        auto start = std::chrono::high_resolution_clock::now();

        // create local buckets
        auto adjusted_desired_bucket_size = static_cast<size_t>(ceil(desired_bucket_size / (double)parallel::total_thread_count() * final_bucket_size_coeff));
        buckets_t<ElementType, KeyFunc, ElementAllocator, BucketType, BucketAllocator> buckets{
            buckets_count,
            adjusted_desired_bucket_size,
            value_range,
            key_func,
            element_allocator,
            bucket_allocator
        };

        // read array part and insert values into local thread buckets
        parallel::for_region([&buckets, first](auto offset)
            {
                buckets.insert(first[offset]);
            }, 0, total_length, total_length_schedule);

        auto end = std::chrono::high_resolution_clock::now();

        local_stats.bucketization = std::chrono::duration_cast<millis_double>(end - start);

        // CONCATENATION / WRITING_BACK

        start = std::chrono::high_resolution_clock::now();

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
        parallel::for_region([advancing_offsets_first, offsets_first](auto offset)
            {
                advancing_offsets_first[offset] = offsets_first[offset];
            }, 0, buckets_count, buckets_count_schedule);

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

        end = std::chrono::high_resolution_clock::now();

        local_stats.concatenation = std::chrono::duration_cast<millis_double>(end - start);
        local_stats.writing_back = std::chrono::duration_cast<millis_double>(end - start);

        return local_stats;
    }

    template<
        typename IORandIt,
        typename KeyFunc,
        typename BOIORandIt,
        typename BSIORandIt
    >
    bucket_sort_measurement bucket_sort_v3_parallel_region_1(
        // shared
        KeyFunc key_func,
        // firstprivate
        IORandIt first,
        size_t buckets_count,
        BOIORandIt bucket_offsets_first,
        BSIORandIt bucket_sizes_first,
        // options
        const parallel::schedule_t& buckets_count_schedule
    )
    {
        bucket_sort_measurement local_stats;

        // SEQUENTIAL_SORTING

        auto start = std::chrono::high_resolution_clock::now();

        parallel::for_region([first, bucket_offsets_first, bucket_sizes_first, &key_func](auto offset)
            {
                auto bucket_first = first + bucket_offsets_first[offset];
                auto bucket_last = bucket_first + bucket_sizes_first[offset];
                std::stable_sort(bucket_first, bucket_last, [&key_func](auto&& l, auto&& r) { return key_func(l) < key_func(r); });
            }, 0, buckets_count, buckets_count_schedule);

        auto end = std::chrono::high_resolution_clock::now();

        local_stats.sequential_sorting = std::chrono::duration_cast<millis_double>(end - start);

        return local_stats;
    }
}

template<
    std::random_access_iterator IORandIt,
    typename KeyFunc = identity_key<typename std::iterator_traits<IORandIt>::value_type>,
    Allocator<typename std::iterator_traits<IORandIt>::value_type> ElementAllocator = std::allocator<typename std::iterator_traits<IORandIt>::value_type>,
    Allocator<size_t> OffsetsAllocator = std::allocator<size_t>,
    Bucket<typename std::iterator_traits<IORandIt>::value_type, ElementAllocator> BucketType = variable_size_bucket_t<typename std::iterator_traits<IORandIt>::value_type, ElementAllocator>,
    Allocator<BucketType> BucketAllocator = std::allocator<BucketType>
>
bucket_sort_measurement bucket_sort_v3(
    IORandIt first, IORandIt last,
    const value_range<std::invoke_result_t<KeyFunc, const typename std::iterator_traits<IORandIt>::value_type&>>& value_range,
    KeyFunc key_func = {},
    size_t desired_bucket_size = 16384,
    double final_bucket_size_coeff = default_bucket_size_coeff,
    ElementAllocator element_allocator = {},
    OffsetsAllocator offsets_allocator = {},
    BucketAllocator bucket_allocator = {},
    const parallel::schedule_t& total_length_schedule = parallel::guided_schedule{.chunk_size=32},
    const parallel::schedule_t& buckets_count_schedule = parallel::guided_schedule{.chunk_size=32},
    std::optional<size_t> num_threads = std::nullopt
)
{
    bucket_sort_measurement total_stats{};

    auto tstart = std::chrono::high_resolution_clock::now();

    auto total_length = std::distance(first, last);
    auto buckets_count = static_cast<size_t>(std::ceil((double)total_length / desired_bucket_size));

    // prepare memory for offset table for each bucket
    fixed_size_dynamic_array<size_t, OffsetsAllocator> bucket_sizes{ buckets_count, static_cast<size_t>(0), offsets_allocator };
    fixed_size_dynamic_array<size_t, OffsetsAllocator> bucket_offsets{ buckets_count, static_cast<size_t>(0), offsets_allocator };
    fixed_size_dynamic_array<size_t, OffsetsAllocator> bucket_advancing_offsets{ buckets_count, static_cast<size_t>(0), offsets_allocator };

    if (num_threads)
    {
        #pragma omp parallel default(none)\
        shared(bucket_sizes, bucket_offsets, bucket_advancing_offsets, value_range, key_func, element_allocator, bucket_allocator, total_length_schedule, buckets_count_schedule, total_stats)\
        firstprivate(first, buckets_count, desired_bucket_size, final_bucket_size_coeff, total_length)\
        num_threads(*num_threads)
        {
            auto local_stats = detail::bucket_sort_v3_parallel_region_0<BucketType>(
                bucket_sizes, bucket_offsets, bucket_advancing_offsets, value_range, key_func, element_allocator, bucket_allocator,
                first, buckets_count, desired_bucket_size, final_bucket_size_coeff, total_length,
                total_length_schedule, buckets_count_schedule
            );
            #pragma omp critical
            total_stats = total_stats + local_stats;
        }
    }
    else
    {
        #pragma omp parallel default(none)\
        shared(bucket_sizes, bucket_offsets, bucket_advancing_offsets, value_range, key_func, element_allocator, bucket_allocator, total_length_schedule, buckets_count_schedule, total_stats)\
        firstprivate(first, buckets_count, desired_bucket_size, final_bucket_size_coeff, total_length)
        {
            auto local_stats = detail::bucket_sort_v3_parallel_region_0<BucketType>(
                bucket_sizes, bucket_offsets, bucket_advancing_offsets, value_range, key_func, element_allocator, bucket_allocator,
                first, buckets_count, desired_bucket_size, final_bucket_size_coeff, total_length,
                total_length_schedule, buckets_count_schedule
            );
            #pragma omp critical
            total_stats = total_stats + local_stats;
        }
    }
    
    // finally sort concatenated buckets across all threads
    auto bucket_offsets_first = std::begin(bucket_offsets);
    auto bucket_sizes_first = std::begin(bucket_sizes);

    if (num_threads)
    {
        #pragma omp parallel default(none)\
        shared(key_func, buckets_count_schedule, total_stats)\
        firstprivate(first, buckets_count, bucket_offsets_first, bucket_sizes_first)\
        num_threads(*num_threads)
        {
            auto local_stats = detail::bucket_sort_v3_parallel_region_1(
                key_func,
                first, buckets_count, bucket_offsets_first, bucket_sizes_first,
                buckets_count_schedule
            );
            #pragma omp critical
            total_stats = total_stats + local_stats;
        }
    }
    else
    {
        #pragma omp parallel default(none)\
        shared(key_func, buckets_count_schedule, total_stats)\
        firstprivate(first, buckets_count, bucket_offsets_first, bucket_sizes_first)
        {
            auto local_stats = detail::bucket_sort_v3_parallel_region_1(
                key_func,
                first, buckets_count, bucket_offsets_first, bucket_sizes_first,
                buckets_count_schedule
            );
            #pragma omp critical
            total_stats = total_stats + local_stats;
        }
    }

    auto tend = std::chrono::high_resolution_clock::now();

    total_stats.total = std::chrono::duration_cast<millis_double>(tend - tstart);
    return total_stats;
}