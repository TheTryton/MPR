#include <parallel_primitives.hpp>
#include <parallel_for.hpp>
#include <time_it.hpp>
#include <buckets.hpp>
#include <bucket_sort.hpp>

#include <memory>
#include <random>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <span>

using namespace parallel;

// GENERATE

template<typename T, typename AllocT>
struct shared_data_std_random
{
    fixed_size_dynamic_array<T, AllocT>& data;
    std::random_device rd;
    value_range<T> value_range;
};

template<typename T>
struct std_random_init_data
{
    std::mt19937 gen;
    std::uniform_real_distribution<T> dist;
};

template<typename T, typename AllocT = std::allocator<T>>
std_random_init_data<T> init_std_random(size_t ti, shared_data_std_random<T, AllocT>& shd)
{
    return {std::mt19937(shd.rd()), std::uniform_real_distribution<T>(shd.value_range.low, shd.value_range.high)};
}

template<typename T, typename AllocT = std::allocator<T>>
void loop_std_random(size_t i, std_random_init_data<T>& init, const shared_data_std_random<T, AllocT>& shd)
{
    shd.data[i] = init.dist(init.gen);
}

template<typename T, typename AllocT = std::allocator<T>>
fixed_size_dynamic_array<T, AllocT> allocate_data(size_t size, AllocT alloc = {})
{
    return fixed_size_dynamic_array<T, AllocT>(size, alloc);
}

template<typename T, typename AllocT>
void generate_data(
    fixed_size_dynamic_array<T, AllocT>& data,
    value_range<T> value_range,
    schedule_t schedule_type,
    std::optional<size_t> thread_count = std::nullopt
    )
{
    shared_data_std_random<T, AllocT> parallel_for_shared_data{
        .data = data,
        .rd = std::random_device(),
        .value_range = value_range
    };

    run(
        data.size(),
        schedule_type,
        parallel_for_shared_data,
        init_std_random<T, AllocT>,
        loop_std_random<T, AllocT>,
        thread_count
    );
}

// GENERATE

template<std::random_access_iterator RandIt>
RandIt unsorted_index(RandIt first, RandIt last)
{
    auto prev = first;
    ++first;
    while (first != last)
    {
        if (*prev > *first)
            return first;
        prev = first;
        ++first;
    }
    return last;
}

template<typename KeyFunc = identity_key<double>, typename ElementAllocator = std::allocator<double>>
void check_sorted(const fixed_size_dynamic_array<double, ElementAllocator>& data, KeyFunc key_func)
{
    auto sorted = std::is_sorted(std::begin(data), std::end(data), [&key_func](auto&& l, auto&& r) { return key_func(l) < key_func(r); });
    
    std::cout << "Check sorted: " << (sorted ? "true" : "false") << std::endl;

    if (!sorted)
    {
        auto it = unsorted_index(std::begin(data), std::end(data));

        std::cout << "First unsorted index: " << (it - std::begin(data)) << std::endl;

        std::cout << "Elements around unsorted index:" << std::endl;
        for (auto first = it - 10; first != it; first++)
        {
            if (first < std::begin(data))
                continue;
            std::cout << *first << ", ";
        }
        std::cout << '[' << *it << ']' << ", ";
        for (auto first = it; first != it + 10; first++)
        {
            if (first >= std::end(data))
                break;
            std::cout << *first << ", ";
        }
    }
}

struct measure_data_point
{
    millis_double allocation;
    millis_double generation;
    bucket_sort_measurement bucket_sort_stats;
    millis_double total;
};

struct measure_stats
{
    mean_stddev allocation;
    mean_stddev generation;
    bucket_sort_stats bucket_sort_stats;
    mean_stddev total;
};

measure_stats stats(const std::vector<measure_data_point>& dps)
{
    return measure_stats{
        .allocation = stats(std::begin(dps), std::end(dps), [](const measure_data_point& dp) {return dp.allocation; }),
        .generation = stats(std::begin(dps), std::end(dps), [](const measure_data_point& dp) {return dp.generation; }),
        .bucket_sort_stats = {
            .bucketization = stats(std::begin(dps), std::end(dps), [](const measure_data_point& dp) {return dp.bucket_sort_stats.bucketization; }),
            .sequential_sorting = stats(std::begin(dps), std::end(dps), [](const measure_data_point& dp) {return dp.bucket_sort_stats.sequential_sorting; }),
            .writing_back = stats(std::begin(dps), std::end(dps), [](const measure_data_point& dp) {return dp.bucket_sort_stats.writing_back; }),
            .concatenation = stats(std::begin(dps), std::end(dps), [](const measure_data_point& dp) {return dp.bucket_sort_stats.concatenation ? *dp.bucket_sort_stats.concatenation : millis_double{}; }),
            .total = stats(std::begin(dps), std::end(dps), [](const measure_data_point& dp) {return dp.bucket_sort_stats.total; }),
        },
        .total = stats(std::begin(dps), std::end(dps), [](const measure_data_point& dp) {return dp.total; }),
    };
}

template<
    typename KeyFunc = identity_key<double>,
    typename ETAlloc = std::allocator<double>,
    typename BucketType = variable_size_bucket_t<double, ETAlloc>,
    typename STAlloc = std::allocator<size_t>,
    typename BTAlloc = std::allocator<BucketType>
>
measure_stats measure_v1(
    size_t length,
    const value_range<double>& v_range,
    KeyFunc key_func = {},
    size_t desired_bucket_size = 16384,
    double final_bucket_size_coeff = default_bucket_size_coeff,
    ETAlloc et_alloc = {},
    STAlloc st_alloc = {},
    BTAlloc bt_alloc = {},
    const parallel::schedule_t& total_length_schedule = parallel::guided_schedule{ .chunk_size = 32 },
    const parallel::schedule_t& buckets_count_schedule = parallel::guided_schedule{ .chunk_size = 32 },
    std::optional<size_t> num_threads = std::nullopt
    )
{
    auto measuredf = [&]() {
        measure_data_point dp;

        auto tstart = std::chrono::high_resolution_clock::now();

        // ALLOCATION

        auto start = std::chrono::high_resolution_clock::now();

        auto data = allocate_data<double>(length, et_alloc);

        auto end = std::chrono::high_resolution_clock::now();
        dp.allocation = std::chrono::duration_cast<millis_double>(end - start);

        // GENERATION

        start = std::chrono::high_resolution_clock::now();

        generate_data(data, v_range, total_length_schedule);

        end = std::chrono::high_resolution_clock::now();
        dp.generation = std::chrono::duration_cast<millis_double>(end - start);

        // SORTING

        dp.bucket_sort_stats = bucket_sort_v1<typename decltype(data)::iterator, KeyFunc, ETAlloc, STAlloc, BucketType>(
            std::begin(data),
            std::end(data),
            v_range.transform(key_func),
            key_func,
            desired_bucket_size,
            final_bucket_size_coeff,
            et_alloc,
            st_alloc,
            bt_alloc,
            num_threads
            );

        auto tend = std::chrono::high_resolution_clock::now();

        dp.total = std::chrono::duration_cast<millis_double>(tend - tstart);

        return dp;
    };

    auto data_points = repeat(measuredf, 10);

    return stats(data_points);
}

template<
    typename KeyFunc = identity_key<double>,
    typename ETAlloc = std::allocator<double>,
    typename BucketType = threadsafe::lockfree_fixed_size_bucket_t<double, ETAlloc>,
    typename STAlloc = std::allocator<size_t>,
    typename BTAlloc = std::allocator<BucketType>
>
auto measure_v2(
    size_t length,
    const value_range<double>& v_range,
    KeyFunc key_func = {},
    size_t desired_bucket_size = 16384,
    double final_bucket_size_coeff = default_bucket_size_coeff,
    ETAlloc et_alloc = {},
    STAlloc st_alloc = {},
    BTAlloc bt_alloc = {},
    const parallel::schedule_t& total_length_schedule = parallel::guided_schedule{ .chunk_size = 32 },
    const parallel::schedule_t& buckets_count_schedule = parallel::guided_schedule{ .chunk_size = 32 },
    std::optional<size_t> num_threads = std::nullopt
)
{
    auto measuredf = [&]() {
        measure_data_point dp;

        auto tstart = std::chrono::high_resolution_clock::now();

        // ALLOCATION

        auto start = std::chrono::high_resolution_clock::now();

        auto data = allocate_data<double>(length, et_alloc);

        auto end = std::chrono::high_resolution_clock::now();
        dp.allocation = std::chrono::duration_cast<millis_double>(end - start);

        // GENERATION

        start = std::chrono::high_resolution_clock::now();

        generate_data(data, v_range, total_length_schedule);

        end = std::chrono::high_resolution_clock::now();
        dp.generation = std::chrono::duration_cast<millis_double>(end - start);

        // SORTING

        dp.bucket_sort_stats = bucket_sort_v2<typename decltype(data)::iterator, KeyFunc, ETAlloc, STAlloc, BucketType>(
            std::begin(data),
            std::end(data),
            v_range.transform(key_func),
            key_func,
            desired_bucket_size,
            final_bucket_size_coeff,
            et_alloc,
            st_alloc,
            bt_alloc,
            total_length_schedule,
            buckets_count_schedule,
            num_threads
            );

        auto tend = std::chrono::high_resolution_clock::now();

        dp.total = std::chrono::duration_cast<millis_double>(tend - tstart);

        return dp;
    };

    auto data_points = repeat(measuredf, 10);

    return stats(data_points);
}

template<
    typename KeyFunc = identity_key<double>,
    typename ETAlloc = std::allocator<double>,
    typename BucketType = variable_size_bucket_t<double, ETAlloc>,
    typename STAlloc = std::allocator<size_t>,
    typename BTAlloc = std::allocator<BucketType>
>
auto measure_v3(
    size_t length,
    const value_range<double>& v_range,
    KeyFunc key_func = {},
    size_t desired_bucket_size = 16384,
    double final_bucket_size_coeff = default_bucket_size_coeff,
    ETAlloc et_alloc = {},
    STAlloc st_alloc = {},
    BTAlloc bt_alloc = {},
    const parallel::schedule_t& total_length_schedule = parallel::guided_schedule{ .chunk_size = 32 },
    const parallel::schedule_t& buckets_count_schedule = parallel::guided_schedule{ .chunk_size = 32 },
    std::optional<size_t> num_threads = std::nullopt
)
{
    auto measuredf = [&]() {
        measure_data_point dp;

        auto tstart = std::chrono::high_resolution_clock::now();

        // ALLOCATION

        auto start = std::chrono::high_resolution_clock::now();

        auto data = allocate_data<double>(length, et_alloc);

        auto end = std::chrono::high_resolution_clock::now();
        dp.allocation = std::chrono::duration_cast<millis_double>(end - start);

        // GENERATION

        start = std::chrono::high_resolution_clock::now();

        generate_data(data, v_range, total_length_schedule);

        end = std::chrono::high_resolution_clock::now();
        dp.generation = std::chrono::duration_cast<millis_double>(end - start);

        // SORTING

        dp.bucket_sort_stats = bucket_sort_v3<typename decltype(data)::iterator, KeyFunc, ETAlloc, STAlloc, BucketType>(
            std::begin(data),
            std::end(data),
            v_range.transform(key_func),
            key_func,
            desired_bucket_size,
            final_bucket_size_coeff,
            et_alloc,
            st_alloc,
            bt_alloc,
            total_length_schedule,
            buckets_count_schedule,
            num_threads
            );

        auto tend = std::chrono::high_resolution_clock::now();

        dp.total = std::chrono::duration_cast<millis_double>(tend - tstart);

        return dp;
    };

    auto data_points = repeat(measuredf, 10);

    auto s = stats(data_points);
    return s;
}

constexpr size_t lengths[] =
{
    1 << 16,
    1 << 18,
    1 << 20,
    1 << 22,
    1 << 24,
};

constexpr size_t desired_bucket_sizes[] =
{
    1 << 4,
    1 << 8,
    1 << 12,
    1 << 16,
};

constexpr double final_bucket_size_coeffs[] =
{
    1.5f,
    2.0f,
    3.0f,
};

constexpr size_t num_threads[] =
{
    1,
    2,
    4,
    8,
    16,
    32,
    64,
};

constexpr parallel::schedule_t total_length_schedule_types[] =
{
    guided_schedule{.chunk_size = 32},
    guided_schedule{.chunk_size = 64},
    guided_schedule{.chunk_size = 128},
};

constexpr parallel::schedule_t buckets_count_schedule_types[] =
{
    guided_schedule{.chunk_size = 4},
    guided_schedule{.chunk_size = 8},
    guided_schedule{.chunk_size = 16},
};

template<template<typename, typename> typename BucketType, typename F>
void call_direct_memcpy(F&& f, size_t length)
{
    std::forward<F>(f)(std::allocator<double>{}, std::allocator<size_t>{}, std::allocator<BucketType<double, std::allocator<double>>>{});
}

template<template<typename, typename> typename BucketType, typename F>
void call_disjoint_prealloc(F&& f, size_t length)
{
    std::pmr::monotonic_buffer_resource bufferet{ length };
    std::pmr::synchronized_pool_resource syncet{ &bufferet };
    std::pmr::polymorphic_allocator<double> etalloc{ &syncet };

    std::pmr::monotonic_buffer_resource bufferst{ length };
    std::pmr::synchronized_pool_resource syncst{ &bufferst };
    std::pmr::polymorphic_allocator<size_t> stalloc{ &syncst };

    std::pmr::monotonic_buffer_resource bufferbt{ length };
    std::pmr::synchronized_pool_resource syncbt{ &bufferbt };
    std::pmr::polymorphic_allocator<BucketType<double, std::pmr::polymorphic_allocator<double>>> btalloc{ &syncbt };

    std::forward<F>(f)(etalloc, stalloc, btalloc);
}

template<template<typename, typename> typename BucketType, typename F>
void call_joint_prealloc(F&& f, size_t length)
{
    std::pmr::monotonic_buffer_resource buffer{ length * 3 };
    std::pmr::synchronized_pool_resource sync{ &buffer };

    std::pmr::polymorphic_allocator<double> etalloc{ &sync };
    std::pmr::polymorphic_allocator<size_t> stalloc{ &sync };
    std::pmr::polymorphic_allocator<BucketType<double, std::pmr::polymorphic_allocator<double>>> btalloc{ &sync };

    std::forward<F>(f)(etalloc, stalloc, btalloc);
}

constexpr std::string_view allocation_presets[]
{
    "direct mempcy",
    "disjoint prealloc",
    "joint prealloc"
};

constexpr auto v_range = value_range<double>{ 0.0f, 1.0f };
constexpr auto key_func = identity_key<>{};

void column_labels(std::ostream& os)
{
    os
        << std::setw(32) << "length" << ", "
        << std::setw(32) << "num_threads" << ", "
        << std::setw(32) << "desired bucket size" << ", "
        << std::setw(32) << "final bucket size coeff" << ", "
        << std::setw(32) << "total length schedule" << ", "
        << std::setw(32) << "buckets count schedule" << ", "
        << std::setw(32) << "allocation preset" << ", "
        << std::setw(32) << "algorithm version" << ", "
        << std::setw(32) << "bucket type" << ", "
        << std::setw(32) << "time allocation (mean)" << ", "
        << std::setw(32) << "time allocation (std)" << ", "
        << std::setw(32) << "time generation (mean)" << ", "
        << std::setw(32) << "time generation (std)" << ", "
        << std::setw(32) << "time bucketization (mean)" << ", "
        << std::setw(32) << "time bucketization (std)" << ", "
        << std::setw(32) << "time sequential sorting (mean)" << ", "
        << std::setw(32) << "time sequential sorting (std)" << ", "
        << std::setw(32) << "time writing back (mean)" << ", "
        << std::setw(32) << "time writing back (std)" << ", "
        << std::setw(32) << "time concatenation (mean)" << ", "
        << std::setw(32) << "time concatenation (std)" << ", "
        << std::setw(32) << "time total sort (mean)" << ", "
        << std::setw(32) << "time total sort (std)" << ", "
        << std::setw(32) << "time total (mean)" << ", "
        << std::setw(32) << "time total (std)"
        << std::endl;
}

void column_values(std::ostream& os,
    size_t length, size_t num_threads,
    size_t desired_bucket_size, double final_bucket_size_coeff,
    parallel::schedule_t tlscht, parallel::schedule_t bcscht,
    std::string_view allocation_preset, size_t algorithm_version,
    std::string_view bucket_type,
    measure_stats mstats)
{
    os
        << std::setw(32) << length << ", "
        << std::setw(32) << num_threads << ", "
        << std::setw(32) << desired_bucket_size << ", "
        << std::setw(32) << final_bucket_size_coeff << ", "
        << std::setw(32) << tlscht << ", "
        << std::setw(32) << bcscht << ", "
        << std::setw(32) << allocation_preset << ", "
        << std::setw(32) << algorithm_version << ", "
        << std::setw(32) << bucket_type << ", "
        << std::setw(32) << mstats.allocation.mean.count() << ", "
        << std::setw(32) << mstats.allocation.stddev.count() << ", "
        << std::setw(32) << mstats.generation.mean.count() << ", "
        << std::setw(32) << mstats.generation.stddev.count() << ", "
        << std::setw(32) << mstats.bucket_sort_stats.bucketization.mean.count() << ", "
        << std::setw(32) << mstats.bucket_sort_stats.bucketization.stddev.count() << ", "
        << std::setw(32) << mstats.bucket_sort_stats.sequential_sorting.mean.count() << ", "
        << std::setw(32) << mstats.bucket_sort_stats.sequential_sorting.stddev.count() << ", "
        << std::setw(32) << mstats.bucket_sort_stats.writing_back.mean.count() << ", "
        << std::setw(32) << mstats.bucket_sort_stats.writing_back.stddev.count() << ", "
        << std::setw(32) << mstats.bucket_sort_stats.concatenation.mean.count() << ", "
        << std::setw(32) << mstats.bucket_sort_stats.concatenation.stddev.count() << ", "
        << std::setw(32) << mstats.bucket_sort_stats.total.mean.count() << ", "
        << std::setw(32) << mstats.bucket_sort_stats.total.stddev.count() << ", "
        << std::setw(32) << mstats.total.mean.count() << ", "
        << std::setw(32) << mstats.total.stddev.count()
        << std::endl;
}

template<typename T, typename TAlloc>
using lockfull_bucket_t = threadsafe::thread_safe_bucket_t<T, TAlloc, variable_size_bucket_t<T, TAlloc>>;
template<typename T, typename TAlloc>
using variable_bucket_t = variable_size_bucket_t<T, TAlloc>;
template<typename T, typename TAlloc>
using lockfree_bucket_t = threadsafe::lockfree_fixed_size_bucket_t<T, TAlloc>;

void measure_all(
    std::ostream& os,
    size_t length, size_t num_threads,
    size_t desired_bucket_size, double final_bucket_size_coeff,
    parallel::schedule_t tlscht, parallel::schedule_t bcscht
)
{
    for (size_t i = 0; i < std::size(allocation_presets); i++)
    {
        switch (i)
        {
        default:
        case 0:
            call_direct_memcpy<variable_bucket_t>(
                [&]<typename ETAlloc, typename STAlloc, typename BTAlloc>(ETAlloc etalloc, STAlloc stalloc, BTAlloc btalloc)
                {
                    auto res = measure_v1<decltype(key_func), ETAlloc, variable_bucket_t<double, ETAlloc>, STAlloc, BTAlloc>(length, v_range, key_func, desired_bucket_size, final_bucket_size_coeff, etalloc, stalloc, btalloc, tlscht, bcscht, num_threads);
                    column_values(os, length, num_threads, desired_bucket_size, final_bucket_size_coeff, tlscht, bcscht, allocation_presets[i], 1, "variable size", res);
                }, length);
            if(final_bucket_size_coeff >= 3.0f)
            {
                call_direct_memcpy<lockfree_bucket_t>(
                    [&]<typename ETAlloc, typename STAlloc, typename BTAlloc>(ETAlloc etalloc, STAlloc stalloc, BTAlloc btalloc)
                    {
                        auto res = measure_v2<decltype(key_func), ETAlloc, lockfree_bucket_t<double, ETAlloc>, STAlloc, BTAlloc>(length, v_range, key_func, desired_bucket_size, final_bucket_size_coeff, etalloc, stalloc, btalloc, tlscht, bcscht, num_threads);
                        column_values(os, length, num_threads, desired_bucket_size, final_bucket_size_coeff, tlscht, bcscht, allocation_presets[i], 2, "fixed size (lockfree)", res);
                    }, length);
            }
            call_direct_memcpy<lockfull_bucket_t>(
                [&]<typename ETAlloc, typename STAlloc, typename BTAlloc>(ETAlloc etalloc, STAlloc stalloc, BTAlloc btalloc)
                {
                    auto res = measure_v2<decltype(key_func), ETAlloc, lockfull_bucket_t<double, ETAlloc>, STAlloc, BTAlloc>(length, v_range, key_func, desired_bucket_size, final_bucket_size_coeff, etalloc, stalloc, btalloc, tlscht, bcscht, num_threads);
                    column_values(os, length, num_threads, desired_bucket_size, final_bucket_size_coeff, tlscht, bcscht, allocation_presets[i], 2, "variable size (lockfull)", res);
                }, length);
            call_direct_memcpy<variable_bucket_t>(
                [&]<typename ETAlloc, typename STAlloc, typename BTAlloc>(ETAlloc etalloc, STAlloc stalloc, BTAlloc btalloc)
                {
                    auto res = measure_v3<decltype(key_func), ETAlloc, variable_bucket_t<double, ETAlloc>, STAlloc, BTAlloc>(length, v_range, key_func, desired_bucket_size, final_bucket_size_coeff, etalloc, stalloc, btalloc, tlscht, bcscht, num_threads);
                    column_values(os, length, num_threads, desired_bucket_size, final_bucket_size_coeff, tlscht, bcscht, allocation_presets[i], 3, "variable size", res);
                }, length);
            break;
        case 1:
            call_disjoint_prealloc<variable_bucket_t>(
                [&]<typename ETAlloc, typename STAlloc, typename BTAlloc>(ETAlloc etalloc, STAlloc stalloc, BTAlloc btalloc)
                {
                    auto res = measure_v1<decltype(key_func), ETAlloc, variable_bucket_t<double, ETAlloc>, STAlloc, BTAlloc>(length, v_range, key_func, desired_bucket_size, final_bucket_size_coeff, etalloc, stalloc, btalloc, tlscht, bcscht, num_threads);
                    column_values(os, length, num_threads, desired_bucket_size, final_bucket_size_coeff, tlscht, bcscht, allocation_presets[i], 1, "variable size", res);
                }, length);
            if(final_bucket_size_coeff >= 3.0f)
            {
                call_disjoint_prealloc<lockfree_bucket_t>(
                    [&]<typename ETAlloc, typename STAlloc, typename BTAlloc>(ETAlloc etalloc, STAlloc stalloc, BTAlloc btalloc)
                    {
                        auto res = measure_v2<decltype(key_func), ETAlloc, lockfree_bucket_t<double, ETAlloc>, STAlloc, BTAlloc>(length, v_range, key_func, desired_bucket_size, final_bucket_size_coeff, etalloc, stalloc, btalloc, tlscht, bcscht, num_threads);
                        column_values(os, length, num_threads, desired_bucket_size, final_bucket_size_coeff, tlscht, bcscht, allocation_presets[i], 2, "fixed size (lockfree)", res);
                    }, length);
            }
            call_disjoint_prealloc<lockfull_bucket_t>(
                [&]<typename ETAlloc, typename STAlloc, typename BTAlloc>(ETAlloc etalloc, STAlloc stalloc, BTAlloc btalloc)
                {
                    auto res = measure_v2<decltype(key_func), ETAlloc, lockfull_bucket_t<double, ETAlloc>, STAlloc, BTAlloc>(length, v_range, key_func, desired_bucket_size, final_bucket_size_coeff, etalloc, stalloc, btalloc, tlscht, bcscht, num_threads);
                    column_values(os, length, num_threads, desired_bucket_size, final_bucket_size_coeff, tlscht, bcscht, allocation_presets[i], 2, "variable size (lockfull)", res);
                }, length);
            call_disjoint_prealloc<variable_bucket_t>(
                [&]<typename ETAlloc, typename STAlloc, typename BTAlloc>(ETAlloc etalloc, STAlloc stalloc, BTAlloc btalloc)
                {
                    auto res = measure_v3<decltype(key_func), ETAlloc, variable_bucket_t<double, ETAlloc>, STAlloc, BTAlloc>(length, v_range, key_func, desired_bucket_size, final_bucket_size_coeff, etalloc, stalloc, btalloc, tlscht, bcscht, num_threads);
                    column_values(os, length, num_threads, desired_bucket_size, final_bucket_size_coeff, tlscht, bcscht, allocation_presets[i], 3, "variable size", res);
                }, length);
            break;
        case 2:
            call_joint_prealloc<variable_bucket_t>(
                [&]<typename ETAlloc, typename STAlloc, typename BTAlloc>(ETAlloc etalloc, STAlloc stalloc, BTAlloc btalloc)
                {
                    auto res = measure_v1<decltype(key_func), ETAlloc, variable_bucket_t<double, ETAlloc>, STAlloc, BTAlloc>(length, v_range, key_func, desired_bucket_size, final_bucket_size_coeff, etalloc, stalloc, btalloc, tlscht, bcscht, num_threads);
                    column_values(os, length, num_threads, desired_bucket_size, final_bucket_size_coeff, tlscht, bcscht, allocation_presets[i], 1, "variable size", res);
                }, length);
            if(final_bucket_size_coeff >= 3.0f)
            {
                call_joint_prealloc<lockfree_bucket_t>(
                    [&]<typename ETAlloc, typename STAlloc, typename BTAlloc>(ETAlloc etalloc, STAlloc stalloc, BTAlloc btalloc)
                    {
                        auto res = measure_v2<decltype(key_func), ETAlloc, lockfree_bucket_t<double, ETAlloc>, STAlloc, BTAlloc>(length, v_range, key_func, desired_bucket_size, final_bucket_size_coeff, etalloc, stalloc, btalloc, tlscht, bcscht, num_threads);
                        column_values(os, length, num_threads, desired_bucket_size, final_bucket_size_coeff, tlscht, bcscht, allocation_presets[i], 2, "fixed size (lockfree)", res);
                    }, length);
            }
            call_joint_prealloc<lockfull_bucket_t>(
                [&]<typename ETAlloc, typename STAlloc, typename BTAlloc>(ETAlloc etalloc, STAlloc stalloc, BTAlloc btalloc)
                {
                    auto res = measure_v2<decltype(key_func), ETAlloc, lockfull_bucket_t<double, ETAlloc>, STAlloc, BTAlloc>(length, v_range, key_func, desired_bucket_size, final_bucket_size_coeff, etalloc, stalloc, btalloc, tlscht, bcscht, num_threads);
                    column_values(os, length, num_threads, desired_bucket_size, final_bucket_size_coeff, tlscht, bcscht, allocation_presets[i], 2, "variable size (lockfull)", res);
                }, length);
            call_joint_prealloc<variable_bucket_t>(
                [&]<typename ETAlloc, typename STAlloc, typename BTAlloc>(ETAlloc etalloc, STAlloc stalloc, BTAlloc btalloc)
                {
                    auto res = measure_v3<decltype(key_func), ETAlloc, variable_bucket_t<double, ETAlloc>, STAlloc, BTAlloc>(length, v_range, key_func, desired_bucket_size, final_bucket_size_coeff, etalloc, stalloc, btalloc, tlscht, bcscht, num_threads);
                    column_values(os, length, num_threads, desired_bucket_size, final_bucket_size_coeff, tlscht, bcscht, allocation_presets[i], 3, "variable size", res);
                }, length);
            break;
        }
    }
}

int main(int argc, char* argv[])
{
    using namespace parallel;

    omp_set_num_threads(16);

    auto selected_os = &std::cout;
    std::ofstream f;

    if (argc >= 2)
    {
        f.open(argv[1]);
        if (f.is_open()) selected_os = &f;
    }

    std::ostream& os = *selected_os;

    column_labels(os);

    for (auto length : lengths)
    {
        std::cout << length << std::endl;
        for (auto nt : num_threads)
            for (auto desired_bucket_size : desired_bucket_sizes)
                for (auto final_bucket_size_coeff : final_bucket_size_coeffs)
                    for (auto tlscht : total_length_schedule_types)
                        for (auto bcscht : buckets_count_schedule_types)
                            measure_all(os, length, nt, desired_bucket_size, final_bucket_size_coeff, tlscht, bcscht);
    }

    return 0;
}