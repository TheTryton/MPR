#include <parallel_primitives.hpp>
#include <parallel_for.hpp>
#include <time_it.hpp>
#include <buckets.hpp>
#include <bucket_sort.hpp>

#include <test_parameters.hpp>

#include <memory>
#include <random>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <span>

using namespace parallel;

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
/*
template<typename T, typename AllocT = std::allocator<T>, typename BT = fixed_bucket_t<T, AllocT>, typename BTAllocT = std::allocator<BT>>
auto create_buckets_factory(size_t bucket_count, double bucket_percentage_size, AllocT alloc = {}, BTAllocT bt_alloc = {})
{
    auto buckets_factory = [bucket_count, bucket_percentage_size, alloc, bt_alloc](size_t data_size, range<T> value_range)
    {
        double bucket_base_size = (double)data_size / (double)bucket_count;
        auto bucket_size = static_cast<size_t>(std::ceil(bucket_base_size * bucket_percentage_size));

        auto bucket_selector = [value_range](const T& value, size_t bucket_count)
        {
            auto coeff01 = (value - value_range.low) / value_range.length();
            return std::pair{static_cast<size_t>(std::round(coeff01 * (bucket_count - 1))), dummy_mutex_t{}};
        };

        auto single_bucket_range_calculator = [value_range, bucket_count](size_t bucket_index)
        {
            auto single_bucket_length = value_range.length() / bucket_count;
            auto single_bucket_low = value_range.low + single_bucket_length * bucket_index;
            auto single_bucket_high = single_bucket_low + single_bucket_length;
            return range<T>{.low = single_bucket_low, .high = single_bucket_high};
        };

        return std::tuple{
            buckets_t<T, AllocT, BT, BTAllocT>{bucket_count, bucket_size, alloc, bt_alloc},
            bucket_selector,
            single_bucket_range_calculator
        };
    };

    return buckets_factory;
}

enum class sorting_order_t
{
    descending,
    ascending,
};

template<typename T, typename BT, typename BucketSelectorF>
void bucketize_v1(
    std::span<T> data,
    size_t thread_starting_offset,
    range<T> thread_value_range,
    BT& buckets,
    BucketSelectorF bucket_selector
    )
{
    size_t data_size = data.size();
    for(size_t i = 0; i < data_size; i++)
    {
        size_t oi = (i + thread_starting_offset) % data_size;
        auto v = data[oi];

        if(thread_value_range.contains(v))
        {
            buckets.insert(v, bucket_selector);
        }
    }
}

template<typename T, typename BucketsFactoryF, typename FallbackSortF>
void bucket_sort_v1(
    std::span<T> data,
    range<T> value_range,
    schedule_t schedule_type,
    size_t single_thread_sort_threshold,
    BucketsFactoryF buckets_factory,
    FallbackSortF fallback_sort,
    sorting_order_t sorting_order,
    size_t thread_count,
    size_t starting_offset
    )
{
    if(data.size() <= 1)
        return;

    auto thread_value_range_length = value_range.length() / thread_count;
    std::mutex save_mutex;
    fixed_size_dynamic_array<size_t> sizes{};
    fixed_size_dynamic_array<size_t> offsets{};

#pragma omp parallel\
    default(none)\
    shared(\
        schedule_type, offsets, sizes, buckets_factory,\
        thread_value_range_length, value_range, data,\
        std::cout, starting_offset, single_thread_sort_threshold,\
        fallback_sort, save_mutex, sorting_order, thread_count)\
    num_threads(thread_count)
    {
#pragma omp single
        {
            sizes = fixed_size_dynamic_array<size_t>(omp_get_num_threads(), 0);
            offsets = fixed_size_dynamic_array<size_t>(omp_get_num_threads(), 0);
        }
#pragma omp barrier

        size_t thread_index = omp_get_thread_num();
        size_t thread_starting_offset = starting_offset * thread_index;

        auto thread_value_range_offset = thread_value_range_length * thread_index;
        auto thread_low = value_range.low + thread_value_range_offset;
        auto thread_high = thread_low + thread_value_range_length;
        auto thread_value_range = range<T>{
            .low = thread_low,
            .high = thread_high
        };

        auto [buckets, bucket_selector, single_bucket_range_calculator] = buckets_factory(data.size(), thread_value_range);

        size_t data_size = data.size();

        // bucketize
        bucketize_v1(data, thread_starting_offset, thread_value_range, buckets, bucket_selector);

        // recursively sort
        size_t bucket_index = 0;
        for(auto& bucket : buckets)
        {
            if(std::size(bucket) <= single_thread_sort_threshold)
            {
                fallback_sort(std::begin(bucket), std::end(bucket));
            }
            else
            {
                bucket_sort_v1(
                    std::span<T>(std::begin(bucket), std::end(bucket)),
                    single_bucket_range_calculator(bucket_index),
                    schedule_type,
                    single_thread_sort_threshold,
                    buckets_factory,
                    fallback_sort,
                    sorting_order,
                    1,
                    starting_offset
                );
                bucket_index++;
                //std::cout << bucket << std::endl;
                //auto it = std::is_sorted_until(std::begin(bucket), std::end(bucket));
                //std::cout << *(it - 1) << ", " << *it << ", " << *(it + 1) << std::endl;
                //std::cout << (it - std::begin(bucket)) << std::endl;
                //std::cout << std::is_sorted(std::begin(bucket), std::end(bucket)) << std::endl;
            }
        }

        //std::stringstream ss;
        //ss << "Thread [" << thread_index << "]:" << '\n';
        //ss << "Range <" << thread_value_range.low << ", " << thread_value_range.high << ")" << '\n';
        //ss << buckets << '\n';
        //std::cout << ss.str() << std::flush;

        // fill total bucket size of thread
        {
            std::lock_guard l{save_mutex};

            sizes[thread_index] = buckets.total_size();
        }
#pragma omp barrier // wait for all threads to finish pushing their total bucket sizes
#pragma omp single // compute offset partial sum
        std::partial_sum(std::begin(sizes), std::end(sizes) - 1, std::begin(offsets) + 1);
#pragma omp barrier
        // copy back to original array with offset to preserve sorting order
        {
            std::lock_guard l{save_mutex};

            auto offset =  offsets[thread_index];
            auto total_size = buckets.total_size();
            auto b = std::begin(data) + offset;
            for(auto& bucket : buckets)
                b = std::copy(std::begin(bucket), std::end(bucket), b);
        }
    }
}


*/
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

void measure_v1(size_t length, const value_range<double>& v_range)
{
    auto schedule_type = guided_schedule{ .chunk_size = 32 };

    auto start = std::chrono::high_resolution_clock::now();

    auto data = allocate_data<double>(length);
    generate_data(data, v_range, schedule_type);
    bucket_sort_v1(std::begin(data), std::end(data), v_range);

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << (end - start).count() / 1e9 << "s" << std::endl;

    auto sorted = std::is_sorted(std::begin(data), std::end(data));
    std::cout << sorted << std::endl;
    if (!sorted)
    {
        auto it = unsorted_index(std::begin(data), std::end(data));
        std::cout << (it - std::begin(data)) << std::endl;
        for (auto first = it - 10; first != it + 10; first++)
        {
            std::cout << *first << ", ";
        }
    }
}

void measure_v2(size_t length, const value_range<double>& v_range)
{
    auto schedule_type = guided_schedule{ .chunk_size = 32 };

    auto start = std::chrono::high_resolution_clock::now();

    auto data = allocate_data<double>(length);
    generate_data(data, v_range, schedule_type);
    bucket_sort_v2(std::begin(data), std::end(data), v_range);

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << (end - start).count() / 1e9 << "s" << std::endl;

    auto sorted = std::is_sorted(std::begin(data), std::end(data));
    std::cout << sorted << std::endl;
    if (!sorted)
    {
        auto it = unsorted_index(std::begin(data), std::end(data));
        std::cout << (it - std::begin(data)) << std::endl;
        for (auto first = it - 10; first != it + 10; first++)
        {
            std::cout << *first << ", ";
        }
    }
}

void measure_v3(size_t length, const value_range<double>& v_range)
{
    auto schedule_type = guided_schedule{ .chunk_size = 32 };

    auto start = std::chrono::high_resolution_clock::now();

    auto data = allocate_data<double>(length);
    generate_data(data, v_range, schedule_type);
    bucket_sort_v3(std::begin(data), std::end(data), v_range);

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << (end - start).count() / 1e9 << "s" << std::endl;

    auto sorted = std::is_sorted(std::begin(data), std::end(data));
    std::cout << sorted << std::endl;
    if (!sorted)
    {
        auto it = unsorted_index(std::begin(data), std::end(data));
        std::cout << (it - std::begin(data)) << std::endl;
        for (auto first = it - 10; first != it + 10; first++)
        {
            std::cout << *first << ", ";
        }
    }
}

int main(int argc, char* argv[])
{
    omp_set_num_threads(16);

    auto length = 10000000;
    auto v_range = value_range<double>{0.0f, 1.0f};
    
    measure_v1(length, v_range);
    measure_v2(length, v_range);
    measure_v3(length, v_range);

    return 0;
}