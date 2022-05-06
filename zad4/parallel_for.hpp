#pragma once

#include <omp.h>

#include <parallel_primitives.hpp>
#include <tuple>

namespace parallel
{
namespace detail
{
template<typename SharedDataT, typename LoopF, typename InitF>
static void visit_schedule_t(const static_schedule& st_sch, std::optional<size_t> n_threads, size_t problem_size, SharedDataT& shared_data, InitF init, LoopF loop)
{
    if(n_threads)
    {
        if(st_sch.chunk_size)
        {
#pragma omp parallel default(none) shared(shared_data, problem_size, init, loop, st_sch) num_threads(*n_threads)
            {
                auto init_data = init(omp_get_thread_num(), shared_data);

#pragma omp for schedule(static, *st_sch.chunk_size)
                for(long i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
            }
        }
        else
        {
#pragma omp parallel default(none) shared(shared_data, problem_size, init, loop) num_threads(*n_threads)
            {
                auto init_data = init(omp_get_thread_num(), shared_data);

#pragma omp for schedule(static)
                for(long i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
            }
        }
    }
    else
    {
        if(st_sch.chunk_size)
        {
#pragma omp parallel default(none) shared(shared_data, problem_size, init, loop, st_sch)
            {
                auto init_data = init(omp_get_thread_num(), shared_data);

#pragma omp for schedule(static, *st_sch.chunk_size)
                for(long i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
            }
        }
        else
        {
#pragma omp parallel default(none) shared(shared_data, problem_size, init, loop)
            {
                auto init_data = init(omp_get_thread_num(), shared_data);

#pragma omp for schedule(static)
                for(long i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
            }
        }
    }
}
template<typename SharedDataT, typename LoopF, typename InitF>
static void visit_schedule_t(const dynamic_schedule& st_sch, std::optional<size_t> n_threads, size_t problem_size, SharedDataT& shared_data, InitF init, LoopF loop)
{
    if(n_threads)
    {
        if(st_sch.chunk_size)
        {
#pragma omp parallel default(none) shared(shared_data, problem_size, init, loop, st_sch) num_threads(*n_threads)
            {
                auto init_data = init(omp_get_thread_num(), shared_data);

#pragma omp for schedule(dynamic, *st_sch.chunk_size)
                for(long i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
            }
        }
        else
        {
#pragma omp parallel default(none) shared(shared_data, problem_size, init, loop) num_threads(*n_threads)
            {
                auto init_data = init(omp_get_thread_num(), shared_data);

#pragma omp for schedule(dynamic)
                for(long i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
            }
        }
    }
    else
    {
        if (st_sch.chunk_size)
        {
#pragma omp parallel default(none) shared(shared_data, problem_size, init, loop, st_sch)
            {
                auto init_data = init(omp_get_thread_num(), shared_data);

#pragma omp for schedule(dynamic, *st_sch.chunk_size)
                for (long i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
            }
        }
        else
        {
#pragma omp parallel default(none) shared(shared_data, problem_size, init, loop)
            {
                auto init_data = init(omp_get_thread_num(), shared_data);

#pragma omp for schedule(dynamic)
                for (long i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
            }
        }
    }
}
template<typename SharedDataT, typename LoopF, typename InitF>
static void visit_schedule_t(const guided_schedule& st_sch, std::optional<size_t> n_threads, size_t problem_size, SharedDataT& shared_data, InitF init, LoopF loop)
{
    if(n_threads)
    {
        if(st_sch.chunk_size)
        {
#pragma omp parallel default(none) shared(shared_data, problem_size, init, loop, st_sch) num_threads(*n_threads)
            {
                auto init_data = init(omp_get_thread_num(), shared_data);

#pragma omp for schedule(guided, *st_sch.chunk_size)
                for(long i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
            }
        }
        else
        {
#pragma omp parallel default(none) shared(shared_data, problem_size, init, loop)  num_threads(*n_threads)
            {
                auto init_data = init(omp_get_thread_num(), shared_data);

#pragma omp for schedule(guided)
                for(long i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
            }
        }
    }
    else
    {
        if(st_sch.chunk_size)
        {
#pragma omp parallel default(none) shared(shared_data, problem_size, init, loop, st_sch)
            {
                auto init_data = init(omp_get_thread_num(), shared_data);

#pragma omp for schedule(guided, *st_sch.chunk_size)
                for(long i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
            }
        }
        else
        {
#pragma omp parallel default(none) shared(shared_data, problem_size, init, loop)
            {
                auto init_data = init(omp_get_thread_num(), shared_data);

#pragma omp for schedule(guided)
                for(long i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
            }
        }
    }
}
template<typename SharedDataT, typename LoopF, typename InitF>
static void visit_schedule_t(const auto_schedule& st_sch, std::optional<size_t> n_threads, size_t problem_size, SharedDataT& shared_data, InitF init, LoopF loop)
{
    if(n_threads)
    {
#pragma omp parallel default(none) shared(shared_data, problem_size, init, loop) num_threads(*n_threads)
        {
            auto init_data = init(omp_get_thread_num(), shared_data);

#ifdef _MSC_VER
#pragma omp for schedule(runtime)
#else
#pragma omp for schedule(auto)
#endif
            for (long i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
        }
    }
    else
    {
#pragma omp parallel default(none) shared(shared_data, problem_size, init, loop)
        {
            auto init_data = init(omp_get_thread_num(), shared_data);
#ifdef _MSC_VER
#pragma omp for schedule(runtime)
#else
#pragma omp for schedule(auto)
#endif
            for (long i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
        }
    }
}
template<typename SharedDataT, typename LoopF, typename InitF>
static void visit_schedule_t(const runtime_schedule& st_sch, std::optional<size_t> n_threads, size_t problem_size, SharedDataT& shared_data, InitF init, LoopF loop)
{
    if(n_threads)
    {
#pragma omp parallel default(none) shared(shared_data, problem_size, init, loop) num_threads(*n_threads)
        {
            auto init_data = init(omp_get_thread_num(), shared_data);

#pragma omp for schedule(runtime)
            for(long i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
        }
    }
    else
    {
#pragma omp parallel default(none) shared(shared_data, problem_size, init, loop)
        {
            auto init_data = init(omp_get_thread_num(), shared_data);

#pragma omp for schedule(runtime)
            for(long i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
        }
    }
}
}

template<typename SharedDataT, typename LoopF, typename InitF>
void run(size_t problem_size, schedule_t schedule_type, SharedDataT& shared_data, InitF init, LoopF loop, std::optional<size_t> n_threads = std::nullopt) noexcept
{
    std::visit([&, problem_size](auto&& sch_t) { detail::visit_schedule_t(sch_t, n_threads, problem_size, shared_data, init, loop); }, schedule_type);
}

namespace detail
{
    // ------------ nowait == true ------------

    template<typename ParallelRegionCode, typename... SharedTypes, typename... FirstPrivateTypes>
    void parallel_region(
        const ParallelRegionCode& parallel_region_code,
        std::tuple<SharedTypes...>&& shared,
        std::tuple<FirstPrivateTypes...> firstprivate,
        std::optional<size_t> num_threads, std::true_type)
    {
        if (num_threads)
        {
#ifndef _MSC_VER
            #pragma omp parallel default(none)\
            shared(shared, parallel_region_code)\
            firstprivate(firstprivate)\
            num_threads(*num_threads)\
            nowait
#else
            #pragma omp parallel default(none)\
            shared(shared, parallel_region_code)\
            firstprivate(firstprivate)\
            num_threads(*num_threads)
#endif
            {
                std::apply(parallel_region_code, std::tuple_cat(shared, std::move(firstprivate)));
            }
        }
        else
        {
#ifndef _MSC_VER
            #pragma omp parallel default(none)\
            shared(shared, parallel_region_code)\
            firstprivate(firstprivate)\
            nowait
#else
            #pragma omp parallel default(none)\
            shared(shared, parallel_region_code)\
            firstprivate(firstprivate)
#endif
            {
                std::apply(parallel_region_code, std::tuple_cat(shared, std::move(firstprivate)));
            }
        }
    }

    // ------------ nowait == false ------------

    // base dispatcher

    template<typename ParallelRegionCode, typename... SharedTypes, typename... FirstPrivateTypes>
    void parallel_region(
        const ParallelRegionCode& parallel_region_code,
        std::tuple<SharedTypes...>&& shared,
        std::tuple<FirstPrivateTypes...> firstprivate,
        std::optional<size_t> num_threads, std::false_type)
    {
        if (num_threads)
        {
            #pragma omp parallel default(none)\
            shared(shared, parallel_region_code)\
            firstprivate(firstprivate)\
            num_threads(*num_threads)
            {
                std::apply(parallel_region_code, std::tuple_cat(shared, std::move(firstprivate)));
            }
        }
        else
        {
            #pragma omp parallel default(none)\
            shared(shared, parallel_region_code)\
            firstprivate(firstprivate)
            {
                std::apply(parallel_region_code, std::tuple_cat(shared, std::move(firstprivate)));
            }
        }
    }
}

template<typename ParallelRegionCode, typename... SharedTypes, typename... FirstPrivateTypes>
void parallel_region(
    const ParallelRegionCode& parallel_region_code,
    std::tuple<SharedTypes...>&& shared,
    std::tuple<FirstPrivateTypes...>&& firstprivate,
    std::optional<size_t> num_threads = std::nullopt, bool nowait = false)
{
    if (nowait)
    {
        return detail::parallel_region(
            std::move(parallel_region_code),
            std::move(shared),
            std::move(firstprivate),
            std::move(num_threads),
            std::true_type{}
        );
    }
    else
    {
        return detail::parallel_region(
            std::move(parallel_region_code),
            std::move(shared),
            std::move(firstprivate),
            std::move(num_threads),
            std::false_type{}
        );
    }
}

namespace detail
{
    // ------------ schedules ------------

    template<typename ForRegionCode>
    void for_region(
        const ForRegionCode& for_region_code,
        index_t init,
        index_t finish,
        const static_schedule& schedule_type
    )
    {
        if (schedule_type.chunk_size)
        {
            #pragma omp for schedule(static, *schedule_type.chunk_size)
            for (index_t i = init; i < finish; ++i)
                for_region_code(i);
        }
        else
        {
            #pragma omp for schedule(static)
            for (index_t i = init; i < finish; ++i)
                for_region_code(i);
        }
    }

    template<typename ForRegionCode>
    void for_region(
        const ForRegionCode& for_region_code,
        index_t init,
        index_t finish,
        const dynamic_schedule& schedule_type
    )
    {
        if (schedule_type.chunk_size)
        {
            #pragma omp for schedule(dynamic, *schedule_type.chunk_size)
            for (index_t i = init; i < finish; ++i)
                for_region_code(i);
        }
        else
        {
            #pragma omp for schedule(dynamic)
            for (index_t i = init; i < finish; ++i)
                for_region_code(i);
        }
    }

    template<typename ForRegionCode>
    void for_region(
        const ForRegionCode& for_region_code,
        index_t init,
        index_t finish,
        const guided_schedule& schedule_type
    )
    {
        if (schedule_type.chunk_size)
        {
            #pragma omp for schedule(guided, *schedule_type.chunk_size)
            for (index_t i = init; i < finish; ++i)
                for_region_code(i);
        }
        else
        {
            #pragma omp for schedule(guided)
            for (index_t i = init; i < finish; ++i)
                for_region_code(i);
        }
    }

    template<typename ForRegionCode>
    void for_region(
        const ForRegionCode& for_region_code,
        index_t init,
        index_t finish,
        const runtime_schedule& schedule_type
    )
    {
        #pragma omp for schedule(runtime)
        for (index_t i = init; i < finish; ++i)
            for_region_code(i);
    }

    template<typename ForRegionCode>
    void for_region(
        const ForRegionCode& for_region_code,
        index_t init,
        index_t finish,
        const auto_schedule& schedule_type
    )
    {
#ifdef _MSC_VER
        #pragma omp for schedule(runtime)
#else
        #pragma omp for schedule(auto)
#endif
        for (index_t i = init; i < finish; ++i)
            for_region_code(i);
    }
}


template<typename ForRegionCode>
void for_region(
    const ForRegionCode& for_region_code,
    index_t init,
    index_t finish,
    const schedule_t& schedule_type
)
{
    return std::visit(
        [&for_region_code, init, finish]<typename T>(T&& schedule_type)
        {
            detail::for_region(for_region_code, init, finish, std::forward<T>(schedule_type));
        }, schedule_type);
}

}