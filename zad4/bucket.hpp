#pragma once

#include <fixed_size_dynamic_array.hpp>

#include <assert.h>

#include <vector>

#include <mutex>

template<typename T, typename AllocT = std::allocator<T>, typename DST = fixed_size_dynamic_array<T, AllocT>>
class fixed_bucket_t
{
public:
    using allocator_type = AllocT;
private:
    DST _data = {};
    size_t _next = 0;
public:
    constexpr fixed_bucket_t() noexcept = default;
    fixed_bucket_t(size_t capacity, AllocT alloc)
        : _data(capacity, std::move(alloc))
    { }
    fixed_bucket_t(const fixed_bucket_t& other) = default;
    fixed_bucket_t(fixed_bucket_t&& other) noexcept = default;
public:
    fixed_bucket_t& operator=(const fixed_bucket_t& other) = default;
    fixed_bucket_t& operator=(fixed_bucket_t&& other) noexcept = default;
public:
    [[nodiscard]] constexpr size_t capacity() const noexcept { return std::size(_data); }
    [[nodiscard]] constexpr size_t size() const noexcept { return _next; }
public:
    constexpr T* begin() noexcept { return std::begin(_data); }
    constexpr const T* begin() const noexcept { return std::begin(_data); }
    constexpr T* end() noexcept { return std::begin(_data) + _next; }
    constexpr const T* end() const noexcept { return std::begin(_data) + _next; }
public:
    constexpr T& at(size_t index) noexcept { return _data[index]; }
    constexpr const T& at(size_t index) const noexcept { return _data[index]; }

    constexpr T& operator[](size_t index) noexcept { return at(index); }
    constexpr const T& operator[](size_t index) const noexcept { return at(index); }
public:
    void insert(const T& value) noexcept
    {
        assert(size() != capacity());
        _data[_next++] = value;
    }
    void insert(T&& value) noexcept
    {
        assert(size() != capacity());
        _data[_next++] = std::move(value);
    }
public:
    template<typename TO, typename AllocTO, typename DSTO>
    friend std::ostream& operator<<(std::ostream& o, const fixed_bucket_t<T, AllocTO, DSTO>& bucket);
};
template<typename T, typename AllocT = std::allocator<T>, typename DST = std::vector<T, AllocT>>
class variable_size_bucket_t
{
public:
    using allocator_type = AllocT;
private:
    DST _data = {};
public:
    constexpr variable_size_bucket_t() noexcept = default;
    variable_size_bucket_t(size_t initial_capacity, AllocT alloc)
        : _data(initial_capacity, std::move(alloc))
    { }
    variable_size_bucket_t(const variable_size_bucket_t& other) = default;
    variable_size_bucket_t(variable_size_bucket_t&& other) noexcept = default;
public:
    variable_size_bucket_t& operator=(const variable_size_bucket_t& other) = default;
    variable_size_bucket_t& operator=(variable_size_bucket_t&& other) noexcept = default;
public:
    [[nodiscard]] constexpr size_t size() const noexcept { return std::size(_data); }
public:
    constexpr auto begin() noexcept { return std::begin(_data); }
    constexpr const auto begin() const noexcept { return std::begin(_data); }
    constexpr auto end() noexcept { return std::end(_data); }
    constexpr const auto end() const noexcept { return std::end(_data); }
public:
    constexpr T& at(size_t index) noexcept { return _data[index]; }
    constexpr const T& at(size_t index) const noexcept { return _data[index]; }

    constexpr T& operator[](size_t index) noexcept { return at(index); }
    constexpr const T& operator[](size_t index) const noexcept { return at(index); }
public:
    void insert(const T& value) noexcept { _data.push_back(value); }
    void insert(T&& value) noexcept { _data.push_back(value); }
public:
    template<typename TO, typename AllocTO, typename DSTO>
    friend std::ostream& operator<<(std::ostream& o, const variable_size_bucket_t<TO, AllocTO, DSTO>& bucket);
};

template<typename T, typename AllocT, typename DST>
std::ostream& operator<<(std::ostream& o, const fixed_bucket_t<T, AllocT, DST>& bucket)
{
    o << "fixed(" << bucket.size() << "/" << bucket.capacity() << ')';
    o << '[';
    for(const auto& v : bucket) o << v << ", ";
    o << ']';
    return o;
}
template<typename T, typename AllocT, typename DST>
std::ostream& operator<<(std::ostream& o, const variable_size_bucket_t<T, AllocT, DST>& bucket)
{
    o << "variable(" << bucket.size() << ')';
    o << '[';
    for(const auto& v : bucket) o << v << ", ";
    o << ']';
    return o;
}

template<typename T, typename LockT = std::mutex, typename AllocT = std::allocator<T>, typename DST = fixed_size_dynamic_array<T, AllocT>, typename BT = fixed_bucket_t<T, AllocT, DST>>
class thread_safe_bucket_t
{
private:
    BT _bucket = {};
    LockT _lock = {};
public:
    constexpr thread_safe_bucket_t() noexcept = default;
    thread_safe_bucket_t(size_t initial_capacity, AllocT alloc)
        : _bucket(initial_capacity, std::move(alloc))
    { }
    thread_safe_bucket_t(const thread_safe_bucket_t& other)
        : _bucket((std::lock_guard{other._lock}, other._bucket))
    { }
    thread_safe_bucket_t(thread_safe_bucket_t&& other)
        : _bucket((std::lock_guard{other._lock}, std::move(other._bucket)))
    { }
public:
    thread_safe_bucket_t& operator=(const thread_safe_bucket_t& other)
    {
        std::scoped_lock{other._lock, _lock};
        _bucket = other._bucket;
        return *this;
    }
    thread_safe_bucket_t& operator=(thread_safe_bucket_t&& other)
    {
        std::scoped_lock{other._lock, _lock};
        _bucket = std::move(other._bucket);
        return *this;
    }
public:
    [[nodiscard]] size_t size() const { return std::size(_bucket); }
public:
    constexpr T* begin() noexcept { return std::begin(_bucket); }
    constexpr const T* begin() const noexcept { return std::begin(_bucket); }
    constexpr T* end() noexcept { return std::end(_bucket); }
    constexpr const T* end() const noexcept { return std::end(_bucket); }
public:
    LockT& lock() noexcept { return _lock; }
    const LockT& lock() const noexcept { return _lock; }
public:
    constexpr T& at(size_t index) noexcept { return _bucket[index]; }
    constexpr const T& at(size_t index) const noexcept { return _bucket[index]; }

    constexpr T& operator[](size_t index) noexcept { return at(index); }
    constexpr const T& operator[](size_t index) const noexcept { return at(index); }
public:
    void insert(const T& value) noexcept
    {
        std::lock_guard{_lock};
        _bucket.insert(value);
    }
    void insert(T&& value) noexcept
    {
        std::lock_guard{_lock};
        _bucket.insert(std::move(value));
    }
public:
    template<typename TO, typename LockTO, typename AllocTO, typename DSTO, typename BTO>
    friend std::ostream& operator<<(std::ostream& o, const thread_safe_bucket_t<TO, LockTO, AllocTO, DSTO, BTO>& bucket);
};

template<typename T, typename LockT, typename AllocT, typename DST, typename BT>
std::ostream& operator<<(std::ostream& o, const thread_safe_bucket_t<T, LockT, AllocT, DST, BT>& bucket)
{
    o << "variable(" << bucket.size() << ')';
    o << '[';
    for(const auto& v : bucket) o << v << ", ";
    o << ']';
    return o;
}
