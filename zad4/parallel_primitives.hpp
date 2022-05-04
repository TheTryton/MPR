#pragma once

#include <omp.h>
#include <optional>
#include <variant>
#include <ostream>
#include <iterator>
#include <numeric>
#include <algorithm>

namespace parallel
{
	using index_t =
#ifdef _MSC_VER
		long;
#else
		size_t;
#endif

    inline index_t current_thread_index()
    {
        return omp_get_thread_num();
    }

    inline index_t total_thread_count()
    {
        return omp_get_num_threads();
    }

    struct static_schedule {
        std::optional<size_t> chunk_size;
        friend std::ostream& operator<<(std::ostream& os, const static_schedule& sch);
    };
    struct dynamic_schedule {
        std::optional<size_t> chunk_size;
        friend std::ostream& operator<<(std::ostream& os, const dynamic_schedule& sch);
    };
    struct guided_schedule {
        std::optional<size_t> chunk_size;
        friend std::ostream& operator<<(std::ostream& os, const guided_schedule& sch);
    };
    struct auto_schedule {
        friend std::ostream& operator<<(std::ostream& os, const auto_schedule& sch);
    };
    struct runtime_schedule {
        friend std::ostream& operator<<(std::ostream& os, const runtime_schedule& sch);
    };

    using schedule_t = std::variant<static_schedule, dynamic_schedule, guided_schedule, auto_schedule, runtime_schedule>;

    std::ostream& operator<<(std::ostream& os, const schedule_t& v);

    namespace algorithm
    {
        // REDUCE

        template<typename BinaryOp, typename It0, typename RT>
        concept ReduceBinaryOp = requires(It0 it0, RT rv, BinaryOp binary_op)
        {
            { binary_op(rv, *it0) } -> std::convertible_to<RT>;
            { binary_op(*it0, rv) } -> std::convertible_to<RT>;
            { binary_op(*it0, *it0) } -> std::convertible_to<RT>;
            { binary_op(rv, rv) } -> std::convertible_to<RT>;
        };

        template<std::random_access_iterator RandIt, class T, ReduceBinaryOp<RandIt, T> BinaryOp>
        constexpr T reduce(
            RandIt first, RandIt last,
            T init,
            BinaryOp binary_op
        )
        {
            if (std::is_constant_evaluated())
            {
                return std::reduce(first, last, init, binary_op);
            }
            else
            {
                T global_best = init;
                auto total_length = std::distance(first, last);

                #pragma omp parallel
                {
                    T local_best = init;
                    auto local_it = first;

                    #pragma omp for nowait
                    for (index_t offset = 0; offset < total_length; ++offset)
                    {
                        local_best = binary_op(local_best, local_it[offset]);
                    }

                    #pragma omp critical
                    {
                        global_best = binary_op(global_best, local_best);
                    }
                }

                return global_best;
            }
        }

        template<std::random_access_iterator RandIt, class T>
        constexpr T reduce(
            RandIt first, RandIt last,
            T init
        )
        {
            return reduce(first, last, init, std::plus<>{});
        }

        template<std::random_access_iterator RandIt>
        requires std::is_default_constructible_v<typename std::iterator_traits<RandIt>::value_type>
        constexpr typename std::iterator_traits<RandIt>::value_type reduce(RandIt first, RandIt last)
        {
            return reduce(first, last, typename std::iterator_traits<RandIt>::value_type{});
        }

        template<
            std::random_access_iterator RandIt,
            ReduceBinaryOp<RandIt, typename std::iterator_traits<RandIt>::value_type> BinaryOp>
        requires std::is_default_constructible_v<typename std::iterator_traits<RandIt>::value_type>
        constexpr typename std::iterator_traits<RandIt>::value_type reduce(RandIt first, RandIt last, BinaryOp binary_op)
        {
            return reduce(first, last, typename std::iterator_traits<RandIt>::value_type{}, binary_op);
        }

        // TRANSFORM

        template<typename UnaryOp, typename It0, typename RT>
        concept TransformUnaryOp = requires(It0 it0, UnaryOp unary_op)
        {
            { unary_op(*it0) } -> std::convertible_to<RT>;
        };

        template<typename BinaryOp, typename It0, typename It1, typename RT>
        concept TransformBinaryOp = requires(It0 it0, It1 it1, BinaryOp unary_op)
        {
            { unary_op(*it0, *it1) } -> std::convertible_to<RT>;
        };

        template<
            std::random_access_iterator InputRandIt,
            std::random_access_iterator OutputRandIt,
            TransformUnaryOp<InputRandIt, OutputRandIt> UnaryOp>
        constexpr void transform(
            InputRandIt first, InputRandIt last,
            OutputRandIt d_first,
            UnaryOp unary_op
        )
        {
            if (std::is_constant_evaluated())
            {
                std::transform(first, last, d_first, unary_op);
            }
            else
            {
                index_t total_length = std::distance(first, last);
                #pragma omp parallel
                {
                    auto local_it = first;
                    auto local_dit = d_first;
                    #pragma omp parallel for
                    for (index_t offset = 0; offset < total_length; ++offset)
                    {
                        local_dit[offset] = unary_op(local_it[offset]);
                    }
                }
            }
        }

        template<
            std::random_access_iterator InputRandIt0,
            std::random_access_iterator InputRandIt1,
            std::random_access_iterator OutputRandIt,
            TransformBinaryOp<InputRandIt0, InputRandIt1, OutputRandIt> BinaryOp>
        constexpr void transform(
            InputRandIt0 first0, InputRandIt0 last0,
            InputRandIt1 first1,
            OutputRandIt d_first,
            BinaryOp binary_op
        )
        {
            if (std::is_constant_evaluated())
            {
                std::transform(first0, last0, first1, d_first, binary_op);
            }
            else
            {
                index_t total_length = std::distance(first0, last0);
                #pragma omp parallel
                {
                    auto local_it0 = first0;
                    auto local_it1 = first1;
                    auto local_dit = d_first;
                    #pragma omp parallel for
                    for (index_t offset = 0; offset < total_length; ++offset)
                    {
                        local_dit[offset] = binary_op(local_it0[offset], local_it1[offset]);
                    }
                }
            }
        }

        // TRANSFORM_REDUCE

        template<typename UnaryOp, typename It0>
        concept TRTransformUnaryOp = requires(It0 it0, UnaryOp unary_op)
        {
            { unary_op(*it0) };
        };

        template<typename BinaryOp, typename It0, typename It1>
        concept TRTransformBinaryOp = requires(It0 it0, It1 it1, BinaryOp binary_op)
        {
            { binary_op(*it0, *it1) };
        };

        template<typename BinaryOp, typename TransformUnaryOp, typename It0, typename RT>
        concept TransformUnaryReduceBinaryOp = requires(It0 it0, RT rv, TransformUnaryOp transform_unary_op, BinaryOp binary_op)
        {
            { binary_op(rv, transform_unary_op(*it0)) } -> std::convertible_to<RT>;
            { binary_op(transform_unary_op(*it0), rv) } -> std::convertible_to<RT>;
            { binary_op(transform_unary_op(*it0), transform_unary_op(*it0)) } -> std::convertible_to<RT>;
            { binary_op(rv, rv) } -> std::convertible_to<RT>;
        };

        template<typename BinaryOp, typename TransformBinaryOp, typename It0, typename It1, typename RT>
        concept TransformBinaryReduceBinaryOp = requires(It0 it0, It1 it1, RT rv, TransformBinaryOp transform_binary_op, BinaryOp binary_op)
        {
            { binary_op(rv, transform_binary_op(*it0, *it1)) } -> std::convertible_to<RT>;
            { binary_op(transform_binary_op(*it0, *it1), rv) } -> std::convertible_to<RT>;
            { binary_op(transform_binary_op(*it0, *it1), transform_binary_op(*it0, *it1)) } -> std::convertible_to<RT>;
            { binary_op(rv, rv) } -> std::convertible_to<RT>;
        };

        // transform binary -> reduce

        template<
            std::random_access_iterator InputRandIt0,
            std::random_access_iterator InputRandIt1,
            class T,
            TRTransformBinaryOp<InputRandIt0, InputRandIt1> TransformBinaryOp,
            TransformBinaryReduceBinaryOp<TransformBinaryOp, InputRandIt0, InputRandIt1, T> ReduceBinaryOp
        >
        constexpr T transform_reduce(
            InputRandIt0 first0, InputRandIt0 last0,
            InputRandIt1 first1,
            T init,
            TransformBinaryOp transform_binary_op,
            ReduceBinaryOp reduce_binary_op
        )
        {
            if (std::is_constant_evaluated())
            {
                return std::transform_reduce(first0, last0, first1, init, transform_binary_op, reduce_binary_op);
            }
            else
            {
                T global_best = init;
                index_t total_length = std::distance(first0, last0);

                #pragma omp parallel
                {
                    auto local_it0 = first0;
                    auto local_it1 = first1;
                    auto local_best = init;

                    #pragma omp for nowait
                    for (index_t offset = 0; offset < total_length; ++offset)
                    {
                        local_best = reduce_binary_op(local_best, transform_binary_op(local_it0[offset], local_it1[offset]));
                    }

                    #pragma omp critical
                    {
                        global_best = reduce_binary_op(global_best, local_best);
                    }
                }

                return global_best;
            }
        }

        template<
            std::random_access_iterator InputRandIt0,
            std::random_access_iterator InputRandIt1,
            class T
        >
        constexpr T transform_reduce(
            InputRandIt0 first0, InputRandIt0 last0,
            InputRandIt1 first1,
            T init
        )
        {
            return transform_reduce(first0, last0, first1, init, std::multiplies<>{}, std::plus<>{});
        }

        template<
            std::random_access_iterator InputRandIt0,
            std::random_access_iterator InputRandIt1
        >
        requires std::is_default_constructible_v<typename std::iterator_traits<InputRandIt0>::value_type>
        constexpr typename std::iterator_traits<InputRandIt0>::value_type transform_reduce(
            InputRandIt0 first0, InputRandIt0 last0,
            InputRandIt1 first1
        )
        {
            return transform_reduce(first0, last0, first1, typename std::iterator_traits<InputRandIt0>::value_type{});
        }

        template<
            std::random_access_iterator InputRandIt0,
            std::random_access_iterator InputRandIt1,
            TRTransformBinaryOp<InputRandIt0, InputRandIt1> TransformBinaryOp,
            TransformBinaryReduceBinaryOp<TransformBinaryOp, InputRandIt0, InputRandIt1, typename std::iterator_traits<InputRandIt0>::value_type> ReduceBinaryOp
        >
        requires std::is_default_constructible_v<typename std::iterator_traits<InputRandIt0>::value_type>
        constexpr typename std::iterator_traits<InputRandIt0>::value_type transform_reduce(
            InputRandIt0 first0, InputRandIt0 last0,
            InputRandIt1 first1,
            TransformBinaryOp transform_binary_op,
            ReduceBinaryOp reduce_binary_op
        )
        {
            return transform_reduce(first0, last0, first1, typename std::iterator_traits<InputRandIt0>::value_type{}, transform_binary_op, reduce_binary_op);
        }

        // transform unary -> reduce

        template<
            std::random_access_iterator InputRandIt0,
            class T,
            TRTransformUnaryOp<InputRandIt0> TransformUnaryOp,
            TransformUnaryReduceBinaryOp<TransformUnaryOp, InputRandIt0, T> ReduceBinaryOp
        >
        constexpr T transform_reduce(
            InputRandIt0 first, InputRandIt0 last,
            T init,
            TransformUnaryOp transform_unary_op,
            ReduceBinaryOp reduce_binary_op
        )
        {
            if (std::is_constant_evaluated())
            {
                return std::transform_reduce(first, last, init, transform_unary_op, reduce_binary_op);
            }
            else
            {
                T global_best = init;
                index_t total_length = std::distance(first, last);

                #pragma omp parallel
                {
                    auto local_it = first;
                    auto local_best = init;

                    #pragma omp for nowait
                    for (index_t offset = 0; offset < total_length; ++offset)
                    {
                        local_best = reduce_binary_op(local_best, transform_unary_op(local_it[offset]));
                    }

                    #pragma omp critical
                    {
                        global_best = reduce_binary_op(global_best, local_best);
                    }
                }

                return global_best;
            }
        }

        template<
            std::random_access_iterator InputRandIt,
            class T
        >
        constexpr T transform_reduce(
            InputRandIt first, InputRandIt last,
            T init
        )
        {
            return transform_reduce(first, last, init, std::multiplies<>{}, std::plus<>{});
        }

        template<
            std::random_access_iterator InputRandIt
        >
        requires std::is_default_constructible_v<typename std::iterator_traits<InputRandIt>::value_type>
        constexpr typename std::iterator_traits<InputRandIt>::value_type transform_reduce(
            InputRandIt first, InputRandIt last
        )
        {
            return transform_reduce(first, last, typename std::iterator_traits<InputRandIt>::value_type{});
        }

        template<
            std::random_access_iterator InputRandIt,
            TRTransformUnaryOp<InputRandIt> TransformUnaryOp,
            TransformUnaryReduceBinaryOp<TransformUnaryOp, InputRandIt, typename std::iterator_traits<InputRandIt>::value_type> ReduceBinaryOp
        >
        requires std::is_default_constructible_v<typename std::iterator_traits<InputRandIt>::value_type>
        constexpr typename std::iterator_traits<InputRandIt>::value_type transform_reduce(
            InputRandIt first, InputRandIt last,
            TransformUnaryOp transform_unary_op,
            ReduceBinaryOp reduce_binary_op
        )
        {
            return transform_reduce(first, last, typename std::iterator_traits<InputRandIt>::value_type{}, transform_unary_op, reduce_binary_op);
        }
    }
}