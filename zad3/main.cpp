#include <omp.h>
#include <new>
#include <memory>
#include <chrono>
#include <iostream>
#include <random>
#include <cstdlib>
#include <optional>
#include <variant>
#include <iomanip>
#include <string>

using std::size_t;
using std::unique_ptr;
using std::make_unique;
using std::literals::operator""s;

struct static_schedule{
    std::optional<size_t> chunk_size;
    friend std::ostream& operator<<(std::ostream& os, const static_schedule& sch);
};
struct dynamic_schedule{
    std::optional<size_t> chunk_size;
    friend std::ostream& operator<<(std::ostream& os, const dynamic_schedule& sch);
};
struct guided_schedule{
    std::optional<size_t> chunk_size;
    friend std::ostream& operator<<(std::ostream& os, const guided_schedule& sch);
};
struct auto_schedule{
    friend std::ostream& operator<<(std::ostream& os, const auto_schedule& sch);
};
struct runtime_schedule{
    friend std::ostream& operator<<(std::ostream& os, const runtime_schedule& sch);
};

std::ostream& operator<<(std::ostream& os, const static_schedule& sch)
{
    os << (sch.chunk_size ? "static["s + std::to_string(*sch.chunk_size) + ']' : "static");
    return os;
}

std::ostream& operator<<(std::ostream& os, const dynamic_schedule& sch)
{
    os << (sch.chunk_size ? "dynamic["s + std::to_string(*sch.chunk_size) + ']' : "dynamic");
    return os;
}

std::ostream& operator<<(std::ostream& os, const guided_schedule& sch)
{
    os << (sch.chunk_size ? "guided["s + std::to_string(*sch.chunk_size) + ']' : "guided");
    return os;
}

std::ostream& operator<<(std::ostream& os, const auto_schedule& sch)
{
    os << "auto";
    return os;
}

std::ostream& operator<<(std::ostream& os, const runtime_schedule& sch)
{
    os << "runtime";
    return os;
}

using schedule_t = std::variant<static_schedule, dynamic_schedule, guided_schedule, auto_schedule, runtime_schedule>;

std::ostream& operator<<(std::ostream& os, const schedule_t& v)
{
    std::visit([&os](auto&& arg) {
        os << arg;
    }, v);
    return os;
}

class parallel_for
{
private:
    schedule_t schedule;
public:
    constexpr parallel_for(schedule_t schedule) noexcept : schedule(schedule) {}
private:
    template<typename SharedDataT, typename LoopF, typename InitF>
    static void visit_schedule_t(const static_schedule& st_sch, size_t problem_size, SharedDataT& shared_data, InitF init, LoopF loop)
    {
        if(st_sch.chunk_size)
        {
#pragma omp parallel default(none) shared(shared_data, problem_size, init, loop, st_sch)
            {
                auto init_data = init(omp_get_thread_num(), shared_data);

#pragma omp for schedule(static, *st_sch.chunk_size)
                for(size_t i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
            }
        }
        else
        {
#pragma omp parallel default(none) shared(shared_data, problem_size, init, loop)
            {
                auto init_data = init(omp_get_thread_num(), shared_data);

#pragma omp for schedule(static)
                for(size_t i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
            }
        }
    }
    template<typename SharedDataT, typename LoopF, typename InitF>
    static void visit_schedule_t(const dynamic_schedule& st_sch, size_t problem_size, SharedDataT& shared_data, InitF init, LoopF loop)
    {
        if(st_sch.chunk_size)
        {
#pragma omp parallel default(none) shared(shared_data, problem_size, init, loop, st_sch)
            {
                auto init_data = init(omp_get_thread_num(), shared_data);

#pragma omp for schedule(dynamic, *st_sch.chunk_size)
                for(size_t i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
            }
        }
        else
        {
#pragma omp parallel default(none) shared(shared_data, problem_size, init, loop)
            {
                auto init_data = init(omp_get_thread_num(), shared_data);

#pragma omp for schedule(dynamic)
                for(size_t i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
            }
        }
    }
    template<typename SharedDataT, typename LoopF, typename InitF>
    static void visit_schedule_t(const guided_schedule& st_sch, size_t problem_size, SharedDataT& shared_data, InitF init, LoopF loop)
    {
        if(st_sch.chunk_size)
        {
#pragma omp parallel default(none) shared(shared_data, problem_size, init, loop, st_sch)
            {
                auto init_data = init(omp_get_thread_num(), shared_data);

#pragma omp for schedule(guided, *st_sch.chunk_size)
                for(size_t i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
            }
        }
        else
        {
#pragma omp parallel default(none) shared(shared_data, problem_size, init, loop)
            {
                auto init_data = init(omp_get_thread_num(), shared_data);

#pragma omp for schedule(guided)
                for(size_t i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
            }
        }
    }
    template<typename SharedDataT, typename LoopF, typename InitF>
    static void visit_schedule_t(const auto_schedule& st_sch, size_t problem_size, SharedDataT& shared_data, InitF init, LoopF loop)
    {
#pragma omp parallel default(none) shared(shared_data, problem_size, init, loop)
        {
            auto init_data = init(omp_get_thread_num(), shared_data);

#pragma omp for schedule(auto)
            for(size_t i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
        }
    }
    template<typename SharedDataT, typename LoopF, typename InitF>
    static void visit_schedule_t(const runtime_schedule& st_sch, size_t problem_size, SharedDataT& shared_data, InitF init, LoopF loop)
    {
#pragma omp parallel default(none) shared(shared_data, problem_size, init, loop)
        {
            auto init_data = init(omp_get_thread_num(), shared_data);

#pragma omp for schedule(runtime)
            for(size_t i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
        }
    }
public:
    template<typename SharedDataT, typename LoopF, typename InitF>
    void run(size_t problem_size, SharedDataT& shared_data, InitF init, LoopF loop) const noexcept
    {
        std::visit([&, problem_size](auto&& sch_t) { visit_schedule_t(sch_t, problem_size, shared_data, init, loop); }, schedule);
    }
    template<typename SharedDataT, typename LoopF, typename InitF>
    void operator()(size_t problem_size, SharedDataT& shared_data, InitF init, LoopF loop) const noexcept
    {
        run(problem_size, shared_data, init, loop);
    }
};

constexpr std::array<schedule_t, 29> schedule_types
{
    static_schedule{},
    static_schedule{.chunk_size = 1},
    static_schedule{.chunk_size = 2},
    static_schedule{.chunk_size = 4},
    static_schedule{.chunk_size = 8},
    static_schedule{.chunk_size = 16},
    static_schedule{.chunk_size = 32},
    static_schedule{.chunk_size = 64},
    static_schedule{.chunk_size = 128},
    dynamic_schedule{},
    dynamic_schedule{.chunk_size = 1},
    dynamic_schedule{.chunk_size = 2},
    dynamic_schedule{.chunk_size = 4},
    dynamic_schedule{.chunk_size = 8},
    dynamic_schedule{.chunk_size = 16},
    dynamic_schedule{.chunk_size = 32},
    dynamic_schedule{.chunk_size = 64},
    dynamic_schedule{.chunk_size = 128},
    guided_schedule{},
    guided_schedule{.chunk_size = 1},
    guided_schedule{.chunk_size = 2},
    guided_schedule{.chunk_size = 4},
    guided_schedule{.chunk_size = 8},
    guided_schedule{.chunk_size = 16},
    guided_schedule{.chunk_size = 32},
    guided_schedule{.chunk_size = 64},
    guided_schedule{.chunk_size = 128},
    auto_schedule{},
    runtime_schedule{},
};

constexpr std::array<size_t, 14> problem_sizes
{
    1ull << 7,
    1ull << 8,
    1ull << 9,
    1ull << 10,
    1ull << 11,
    1ull << 12,
    1ull << 13,
    1ull << 14,
    1ull << 15,
    1ull << 16,
    1ull << 18,
    1ull << 20,
    1ull << 22,
    1ull << 24,
};

template<typename F>
std::pair<std::chrono::duration<double, std::milli>, std::chrono::duration<double, std::milli>>
time_it(F f, size_t repetitions = 1000)
{
    using millis_double = std::chrono::duration<double, std::milli>;
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
                    millis_double(), std::plus<millis_double>()
            ) / repetitions;

    std::transform(std::begin(data_points), std::end(data_points), std::begin(data_points), [&](auto&& dp){
        auto diff = mean - dp;
        return millis_double(diff.count() * diff.count());
    });

    auto variance =
            std::accumulate(
                    std::begin(data_points), std::end(data_points),
                    millis_double(), std::plus<millis_double>()
            ) / repetitions;

    return {mean, millis_double(sqrt(variance.count()))};
}

template<typename T>
struct shared_data_crand
{
    std::unique_ptr<T[]> data;
};

template<typename T>
struct shared_data_std_random
{
    std::unique_ptr<T[]> data;
    std::random_device rd;
};

struct crand_init_data_t{};

template<typename T>
crand_init_data_t init_crand(int ti, const shared_data_crand<T>& shd) { return {}; } //noop

template<typename T>
void loop_crand(int i, const crand_init_data_t& init, const shared_data_crand<T>& shd)
{
    shd.data[i] = rand();
}

struct std_random_init_data
{
    std::mt19937 gen;
    std::uniform_int_distribution<> dist;
};

template<typename T>
std_random_init_data init_std_random(int ti, shared_data_std_random<T>& shd)
{
    return {std::mt19937(shd.rd()), std::uniform_int_distribution<>(1, 1000)};
}

template<typename T>
void loop_std_random(int i, std_random_init_data& init, const shared_data_std_random<T>& shd)
{
    shd.data[i] = init.dist(init.gen);
}

void run_all_crand()
{
    using std::cout;
    using std::cin;
    using std::endl;

    using seconds_double_t = std::chrono::duration<double, std::ratio<1, 1>>;

    cout <<
         std::left << std::setw(20) << "Problem Size" << "," <<
         std::left << std::setw(20) << "Schedule Type" << "," <<
         std::left << std::setw(20) << "Time Taken (Mean) [s]" <<
         std::left << std::setw(20) << "Time Taken STD [s]" <<
         endl;

    for(auto problem_size : problem_sizes)
    {
        for(auto schedule_type : schedule_types)
        {
            shared_data_crand<int> shd{
                    .data = std::make_unique<int[]>(problem_size)
            };
            auto pf = parallel_for(schedule_type);

            auto [mean, std] = time_it([&](){pf.run(problem_size, shd, init_crand<int>, loop_crand<int>);}, 100);

            cout <<
                 std::left << std::setw(20) << problem_size << "," <<
                 std::left << std::setw(20) << schedule_type << "," <<
                 std::left << std::setw(20) << std::chrono::duration_cast<seconds_double_t>(mean).count() <<
                 std::left << std::setw(20) << std::chrono::duration_cast<seconds_double_t>(std).count() <<
                 endl;
        }
    }
}

void run_all_std_random()
{
    using std::cout;
    using std::cin;
    using std::endl;

    using seconds_double_t = std::chrono::duration<double, std::ratio<1, 1>>;

    cout <<
        std::left << std::setw(20) << "Problem Size" << "," <<
        std::left << std::setw(20) << "Schedule Type" << "," <<
        std::left << std::setw(20) << "Time Taken (Mean) [s]" << "," <<
        std::left << std::setw(20) << "Time Taken STD [s]" <<
    endl;

    for(auto problem_size : problem_sizes)
    {
        for(auto schedule_type : schedule_types)
        {
            shared_data_std_random<int> shd{
                .data = std::make_unique<int[]>(problem_size),
                .rd = std::random_device()
            };
            auto pf = parallel_for(schedule_type);

            auto [mean, std] = time_it([&](){pf.run(problem_size, shd, init_std_random<int>, loop_std_random<int>);}, 100);

            cout <<
                 std::left << std::setw(20) << problem_size << "," <<
                 std::left << std::setw(20) << schedule_type << "," <<
                 std::left << std::setw(20) << std::chrono::duration_cast<seconds_double_t>(mean).count() << "," <<
                 std::left << std::setw(20) << std::chrono::duration_cast<seconds_double_t>(std).count() <<
            endl;
        }
    }
}

int main(int argc, char* argv[])
{
    omp_set_num_threads(16);

    run_all_std_random();

    return 0;
}