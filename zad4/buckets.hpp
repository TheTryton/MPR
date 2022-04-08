#pragma once

#include <bucket.hpp>
#include <lockables.hpp>

#include <range.hpp>

template<typename T, typename AllocT = std::allocator<T>, typename BT = fixed_bucket_t<T, AllocT>, typename BTAllocT = std::allocator<BT>>
class buckets_t
{
private:
    fixed_size_dynamic_array<BT, BTAllocT> _buckets = {};
    range<T> _values_range = {};
public:
    constexpr buckets_t() noexcept = default;
    buckets_t(
        size_t bucket_count, size_t bucket_size, const range<T>& inserted_values_range,
        AllocT alloc = {}, BTAllocT bt_alloc = {}
    )
        : _buckets(bucket_count, BT{bucket_size, std::move(alloc)}, std::move(bt_alloc))
        , _values_range(inserted_values_range)
    {}
    buckets_t(const buckets_t& other) = default;
    buckets_t(buckets_t&& other) noexcept = default;
public:
    buckets_t& operator=(const buckets_t& other) = default;
    buckets_t& operator=(buckets_t&& other) noexcept = default;
public:
    [[nodiscard]] constexpr size_t size() const noexcept { return std::size(_buckets); }
    [[nodiscard]] constexpr size_t total_size() const noexcept
    {
        return std::accumulate(std::begin(_buckets), std::end(_buckets), size_t(0), [](auto a, auto b){ return a + std::size(b);});
    }
public:
    constexpr BT* begin()  noexcept { return std::begin(_buckets); }
    constexpr const BT* begin() const noexcept { return std::begin(_buckets); }
    constexpr BT* end() noexcept { return std::end(_buckets); }
    constexpr const BT* end() const noexcept { return std::end(_buckets); }
public:
    constexpr BT& at(size_t bucket_index) noexcept { return _buckets[bucket_index]; }
    constexpr const BT& at(size_t bucket_index) const noexcept { return _buckets[bucket_index]; }

    constexpr BT& operator[](size_t bucket_index) noexcept { return at(bucket_index); }
    constexpr const BT& operator[](size_t bucket_index) const noexcept { return at(bucket_index); }
private:
    constexpr size_t select_bucket(const T& value) const noexcept
    {
        assert(can_accept_value(value));
        auto coeff01 = (value - _values_range.low) / _values_range.length();
        return static_cast<size_t>(std::round(coeff01 * (size() - 1)));
    }
public:
    constexpr bool can_accept_value(const T& value) const noexcept
    {
        return _values_range.contains(value);
    }
    constexpr range<T> bucket_value_range(size_t bucket_index) const noexcept
    {
        return _values_range.split_index(size(), bucket_index);
    }
public:
    void insert(const T& value) noexcept
    {
        _buckets[select_bucket(value)].insert(value);
    }
    void insert(T&& value) noexcept
    {
        _buckets[select_bucket(value)].insert(std::move(value));
    }
public:
    template<typename TO, typename AllocTO, typename BTO, typename BTAllocTO>
    friend std::ostream& operator<<(std::ostream& o, const buckets_t<TO, AllocTO, BTO, BTAllocTO>& buckets);
};

template<typename T, typename AllocT, typename BT, typename BTAllocT>
std::ostream& operator<<(std::ostream& o, const buckets_t<T, AllocT, BT, BTAllocT>& buckets)
{
    size_t i = 0;
    for(const auto& bucket : buckets)
    {
        o << '{' << i++ << "} -> " << bucket << '\n';
    }
    return o;
}

template<typename T, typename LockT = std::mutex, typename AllocT = std::allocator<T>, typename BT = fixed_bucket_t<T, AllocT>, typename BTAllocT = std::allocator<BT>>
class thread_safe_buckets_t
{
private:
    buckets_t<T, AllocT, BT, BTAllocT> _buckets = {};
    LockT _lock = {};
public:
    constexpr thread_safe_buckets_t() noexcept = default;
    thread_safe_buckets_t(
        size_t bucket_count, size_t bucket_size, const range<T>& inserted_values_range,
        AllocT alloc = {}, BTAllocT bt_alloc = {}
    )
        : _buckets(bucket_count, bucket_size, inserted_values_range, std::move(alloc), std::move(bt_alloc))
    { }
    thread_safe_buckets_t(const thread_safe_buckets_t& other)
        : _buckets((std::lock_guard{other._lock}, other._buckets))
    { }
    thread_safe_buckets_t(thread_safe_buckets_t&& other)
        : _buckets((std::lock_guard{other._lock}, other._buckets))
    { }
public:
    thread_safe_buckets_t& operator=(const thread_safe_buckets_t& other)
    {
        std::scoped_lock{other._lock, _lock};
        _buckets = other._buckets;
        return *this;
    }
    thread_safe_buckets_t& operator=(thread_safe_buckets_t&& other)
    {
        std::scoped_lock{other._lock, _lock};
        _buckets = std::move(other._buckets);
        return *this;
    }
public:
    [[nodiscard]] constexpr size_t size() const noexcept { return std::size(_buckets); }
    [[nodiscard]] constexpr size_t total_size() const noexcept
    {
        return std::accumulate(std::begin(_buckets), std::end(_buckets), size_t(0), [](auto a, auto b){ return a + std::size(b);});
    }
public:
    constexpr BT* begin()  noexcept { return std::begin(_buckets); }
    constexpr const BT* begin() const noexcept { return std::begin(_buckets); }
    constexpr BT* end() noexcept { return std::end(_buckets); }
    constexpr const BT* end() const noexcept { return std::end(_buckets); }
public:
    LockT& lock() noexcept { return _lock; }
    const LockT& lock() const noexcept { return _lock; }
public:
    constexpr BT& at(size_t bucket_index) noexcept { return _buckets[bucket_index]; }
    constexpr const BT& at(size_t bucket_index) const noexcept { return _buckets[bucket_index]; }

    constexpr BT& operator[](size_t bucket_index) noexcept { return at(bucket_index); }
    constexpr const BT& operator[](size_t bucket_index) const noexcept { return at(bucket_index); }
public:
    constexpr bool can_accept_value(const T& value) const noexcept
    {
        return _buckets.can_accept_value(value);
    }
    constexpr range<T> bucket_value_range(size_t bucket_index) const noexcept
    {
        return _buckets.bucket_value_range(bucket_index);
    }
public:
    void insert(const T& value) noexcept
    {
        std::lock_guard{_lock};
        _buckets[select_bucket(value)].insert(value);
    }
    void insert(T&& value) noexcept
    {
        std::lock_guard{_lock};
        _buckets[select_bucket(value)].insert(std::move(value));
    }
public:
    template<typename TO, typename LockTO, typename AllocTO, typename BTO, typename BTAllocTO>
    friend std::ostream& operator<<(std::ostream& o, const thread_safe_buckets_t<TO, LockTO, AllocTO, BTO, BTAllocTO>& buckets);
};

template<typename T, typename LockT, typename AllocT, typename BT, typename BTAllocT>
std::ostream& operator<<(std::ostream& o, const thread_safe_buckets_t<T, LockT, AllocT, BT, BTAllocT>& buckets)
{
    size_t i = 0;
    for(const auto& bucket : buckets)
    {
        o << '{' << i++ << "} -> " << bucket << '\n';
    }
    return o;
}
