#pragma once

#include <chrono>
#include <vector>
#include <numeric>
#include <cmath>

using millis_double = std::chrono::duration<double, std::milli>;
using seconds_double = std::chrono::duration<double, std::ratio<1,1>>;

struct time_it_result
{
    millis_double mean;
    millis_double stddev;
};

template<typename F>
time_it_result time_it(F f, size_t repetitions = 1000)
{
    std::vector<millis_double> data_points(repetitions);

    for(size_t i=0; i<repetitions; i++)
    {
        auto stddev_start = std::chrono::high_resolution_clock::now();
        f();
        auto stddev_end = std::chrono::high_resolution_clock::now();
        data_points[i] = std::chrono::duration_cast<millis_double>(stddev_end - stddev_start);
    }

    auto mean =
        std::accumulate(
            std::begin(data_points), std::end(data_points),
            millis_double(), std::plus<>()
        ) / repetitions;

    std::transform(std::begin(data_points), std::end(data_points), std::begin(data_points), [&](auto&& dp){
        auto diff = mean - dp;
        return millis_double(diff.count() * diff.count());
    });

    auto variance =
        std::accumulate(
            std::begin(data_points), std::end(data_points),
            millis_double(), std::plus<>()
        ) / repetitions;

    return {mean, millis_double(sqrt(variance.count()))};
}
