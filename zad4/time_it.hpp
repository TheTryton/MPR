#pragma once

#include <chrono>
#include <vector>
#include <numeric>
#include <cmath>

using millis_double = std::chrono::duration<double, std::milli>;
using seconds_double = std::chrono::duration<double, std::ratio<1,1>>;

struct mean_stddev
{
    millis_double mean;
    millis_double stddev;
};

template<typename It, typename F>
mean_stddev stats(It b, It e, F f = [](auto&& v) {return v; })
{
    auto mean =
        std::accumulate(
            b, e,
            millis_double(), [&f](auto&& a, auto&& b) {return a + f(b); }
        ) / std::distance(b, e);

    auto variance =
        std::transform_reduce(
            b, e,
            millis_double(),
            std::plus<>(),
            [&](auto&& dp) {
                auto diff = mean - f(dp);
                return millis_double(diff.count() * diff.count());
            }
        ) / std::distance(b, e);

    return mean_stddev{.mean = mean, .stddev = millis_double(sqrt(variance.count()))};
}

template<typename F>
std::vector<std::invoke_result_t<F>> repeat(F f, size_t repetitions = 1000)
{
    std::vector<std::invoke_result_t<F>> data_points(repetitions);

    std::generate_n(std::begin(data_points), repetitions, f);

    return data_points;
}

template<typename F>
mean_stddev time_it(F f, size_t repetitions = 1000)
{
    auto data_points = repeat([&f]() {
        auto start = std::chrono::high_resolution_clock::now();
        f();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<millis_double>(end - start);
        }, repetitions);

    return stats(std::begin(data_points), std::end(data_points));
}
