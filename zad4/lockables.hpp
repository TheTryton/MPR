#pragma once

class dummy_mutex_t
{
public:
    constexpr void lock() const noexcept {};
    constexpr void unlock() const noexcept {};
};
