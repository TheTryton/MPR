#pragma once

#include <omp.h>

#include <schedule.hpp>

namespace parallel_for
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
                for(size_t i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
            }
        }
        else
        {
#pragma omp parallel default(none) shared(shared_data, problem_size, init, loop) num_threads(*n_threads)
            {
                auto init_data = init(omp_get_thread_num(), shared_data);

#pragma omp for schedule(static)
                for(size_t i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
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
                for(size_t i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
            }
        }
        else
        {
#pragma omp parallel default(none) shared(shared_data, problem_size, init, loop) num_threads(*n_threads)
            {
                auto init_data = init(omp_get_thread_num(), shared_data);

#pragma omp for schedule(dynamic)
                for(size_t i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
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
                for (size_t i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
            }
        }
        else
        {
#pragma omp parallel default(none) shared(shared_data, problem_size, init, loop)
            {
                auto init_data = init(omp_get_thread_num(), shared_data);

#pragma omp for schedule(dynamic)
                for (size_t i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
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
                for(size_t i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
            }
        }
        else
        {
#pragma omp parallel default(none) shared(shared_data, problem_size, init, loop)  num_threads(*n_threads)
            {
                auto init_data = init(omp_get_thread_num(), shared_data);

#pragma omp for schedule(guided)
                for(size_t i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
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
}
template<typename SharedDataT, typename LoopF, typename InitF>
static void visit_schedule_t(const auto_schedule& st_sch, std::optional<size_t> n_threads, size_t problem_size, SharedDataT& shared_data, InitF init, LoopF loop)
{
    if(n_threads)
    {
#pragma omp parallel default(none) shared(shared_data, problem_size, init, loop) num_threads(*n_threads)
        {
            auto init_data = init(omp_get_thread_num(), shared_data);

#pragma omp for schedule(auto)
            for (size_t i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
        }
    }
    else
    {
#pragma omp parallel default(none) shared(shared_data, problem_size, init, loop)
        {
            auto init_data = init(omp_get_thread_num(), shared_data);

#pragma omp for schedule(auto)
            for (size_t i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
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
            for(size_t i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
        }
    }
    else
    {
#pragma omp parallel default(none) shared(shared_data, problem_size, init, loop)
        {
            auto init_data = init(omp_get_thread_num(), shared_data);

#pragma omp for schedule(runtime)
            for(size_t i = 0; i < problem_size; i++) loop(i, init_data, shared_data);
        }
    }
}
}

template<typename SharedDataT, typename LoopF, typename InitF>
void run(size_t problem_size, schedule_t schedule_type, SharedDataT& shared_data, InitF init, LoopF loop, std::optional<size_t> n_threads = std::nullopt) noexcept
{
    std::visit([&, problem_size](auto&& sch_t) { detail::visit_schedule_t(sch_t, n_threads, problem_size, shared_data, init, loop); }, schedule_type);
}
}
