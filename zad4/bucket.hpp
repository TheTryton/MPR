#pragma once

#include <fixed_size_dynamic_array.hpp>

#include <assert.h>
#include <vector>
#include <mutex>
#include <atomic>

template<
    typename ElementType,
    typename ElementAllocator = std::allocator<ElementType>,
    typename FixedStorageType = fixed_size_dynamic_array<ElementType, ElementAllocator>
    >
class fixed_size_bucket_t
{
public:
    using element_allocator_t = ElementAllocator;
    using storage_type = FixedStorageType;
public:
    using value_type = ElementType;
    using reference = value_type&;
    using const_reference = const value_type&;

    using iterator = typename storage_type::iterator;
    using const_iterator = typename storage_type::const_iterator;

    using size_type = typename storage_type::size_type;
private:
    storage_type _data = {};
    size_type _next = 0;
public:
    constexpr fixed_size_bucket_t() noexcept = default;
    fixed_size_bucket_t(size_type capacity, element_allocator_t alloc)
        : _data(capacity, std::move(alloc))
    { }
    fixed_size_bucket_t(const fixed_size_bucket_t& other) = default;
    fixed_size_bucket_t(fixed_size_bucket_t&& other) noexcept = default;
public:
    fixed_size_bucket_t& operator=(const fixed_size_bucket_t& other) = default;
    fixed_size_bucket_t& operator=(fixed_size_bucket_t&& other) noexcept = default;
public:
    [[nodiscard]] constexpr size_type capacity() const noexcept { return std::size(_data); }
    [[nodiscard]] constexpr size_type size() const noexcept { return _next; }
public:
    constexpr iterator begin() noexcept { return std::begin(_data); }
    constexpr const_iterator begin() const noexcept { return std::begin(_data); }
    constexpr iterator end() noexcept { return std::begin(_data) + _next; }
    constexpr const_iterator end() const noexcept { return std::begin(_data) + _next; }
public:
    constexpr reference at(size_type index) noexcept { return _data[index]; }
    constexpr const_reference at(size_type index) const noexcept { return _data[index]; }

    constexpr reference operator[](size_type index) noexcept { return at(index); }
    constexpr const_reference operator[](size_type index) const noexcept { return at(index); }
public:
    void insert(const_reference value) noexcept
    {
        assert(size() != capacity());
        _data[_next++] = value;
    }
    void insert(value_type&& value) noexcept
    {
        assert(size() != capacity());
        _data[_next++] = std::move(value);
    }
public:
    template<
        typename ElementTypeO,
        typename ElementAllocatorO,
        typename FixedStorageTypeO
        >
    friend std::ostream& operator<<(
        std::ostream& o,
        const fixed_size_bucket_t<ElementTypeO, ElementAllocatorO, FixedStorageTypeO>& bucket
        );
};
template<
    typename ElementType,
    typename ElementAllocator = std::allocator<ElementType>,
    typename VariableStorageType = std::vector<ElementType, ElementAllocator>
    >
class variable_size_bucket_t
{
public:
    using element_allocator_t = ElementAllocator;
    using storage_type = VariableStorageType;
public:
    using value_type = ElementType;
    using reference = value_type&;
    using const_reference = const value_type&;

    using iterator = typename storage_type::iterator;
    using const_iterator = typename storage_type::const_iterator;

    using size_type = typename storage_type::size_type;
private:
    storage_type _data = {};
public:
    constexpr variable_size_bucket_t() noexcept = default;
    variable_size_bucket_t(size_type initial_capacity, ElementAllocator alloc)
        : _data(std::move(alloc))
    {
        _data.reserve(initial_capacity);
    }
    variable_size_bucket_t(const variable_size_bucket_t& other) = default;
    variable_size_bucket_t(variable_size_bucket_t&& other) noexcept = default;
public:
    variable_size_bucket_t& operator=(const variable_size_bucket_t& other) = default;
    variable_size_bucket_t& operator=(variable_size_bucket_t&& other) noexcept = default;
public:
    [[nodiscard]] constexpr size_t size() const noexcept { return std::size(_data); }
public:
    constexpr iterator begin() noexcept { return std::begin(_data); }
    constexpr const_iterator begin() const noexcept { return std::begin(_data); }
    constexpr iterator end() noexcept { return std::end(_data); }
    constexpr const_iterator end() const noexcept { return std::end(_data); }
public:
    constexpr reference at(size_t index) noexcept { return _data[index]; }
    constexpr const_reference at(size_t index) const noexcept { return _data[index]; }

    constexpr reference operator[](size_t index) noexcept { return at(index); }
    constexpr const_reference operator[](size_t index) const noexcept { return at(index); }
public:
    void insert(const_reference value) noexcept { _data.push_back(value); }
    void insert(value_type&& value) noexcept { _data.push_back(value); }
public:
    template<
        typename ElementTypeO,
        typename ElementAllocatorO,
        typename VariableStorageTypeO
        >
    friend std::ostream& operator<<(
        std::ostream& o,
        const variable_size_bucket_t<ElementTypeO, ElementAllocatorO, VariableStorageTypeO>& bucket
        );
};

template<
    typename ElementType,
    typename ElementAllocator,
    typename FixedStorageType
        >
std::ostream& operator<<(
    std::ostream& o,
    const fixed_size_bucket_t<ElementType, ElementAllocator, FixedStorageType>& bucket
)
{
    o << "fixed(" << bucket.size() << "/" << bucket.capacity() << ')';
    o << '[';
    for(const auto& v : bucket) o << v << ", ";
    o << ']';
    return o;
}
template<
    typename ElementType,
    typename ElementAllocator,
    typename VariableStorageType
    >
std::ostream& operator<<(
    std::ostream& o,
    const variable_size_bucket_t<ElementType, ElementAllocator, VariableStorageType>& bucket
)
{
    o << "variable(" << bucket.size() << ')';
    o << '[';
    for(const auto& v : bucket) o << v << ", ";
    o << ']';
    return o;
}

namespace threadsafe
{
template<
    typename ElementType,
    typename ElementAllocator = std::allocator<ElementType>,
    typename FixedStorageType = fixed_size_dynamic_array<ElementType, ElementAllocator>
    >
class lockfree_fixed_size_bucket_t
{
public:
    using element_allocator_t = ElementAllocator;
    using storage_type = FixedStorageType;
public:
    using value_type = ElementType;
    using reference = value_type&;
    using const_reference = const value_type&;

    using iterator = typename storage_type::iterator;
    using const_iterator = typename storage_type::const_iterator;

    using size_type = typename storage_type::size_type;
private:
    storage_type _data = {};
    std::atomic<size_type> _next = 0;
public:
    constexpr lockfree_fixed_size_bucket_t() noexcept = default;
    lockfree_fixed_size_bucket_t(size_type capacity, element_allocator_t alloc)
        : _data(capacity, std::move(alloc))
    { }
    lockfree_fixed_size_bucket_t(const lockfree_fixed_size_bucket_t& other) = default;
    lockfree_fixed_size_bucket_t(lockfree_fixed_size_bucket_t&& other) noexcept = default;
public:
    lockfree_fixed_size_bucket_t& operator=(const lockfree_fixed_size_bucket_t& other) = default;
    lockfree_fixed_size_bucket_t& operator=(lockfree_fixed_size_bucket_t&& other) noexcept = default;
public:
    [[nodiscard]] constexpr size_type capacity() const noexcept { return std::size(_data); }
    [[nodiscard]] constexpr size_type size() const noexcept { return _next; }
public:
    constexpr iterator begin() noexcept { return std::begin(_data); }
    constexpr const_iterator begin() const noexcept { return std::begin(_data); }
    constexpr iterator end() noexcept { return std::begin(_data) + _next; }
    constexpr const_iterator end() const noexcept { return std::begin(_data) + _next; }
public:
    constexpr reference at(size_type index) noexcept { return _data[index]; }
    constexpr const_reference at(size_type index) const noexcept { return _data[index]; }

    constexpr reference operator[](size_type index) noexcept { return at(index); }
    constexpr const_reference operator[](size_type index) const noexcept { return at(index); }
public:
    void insert(const_reference value) noexcept
    {
        assert(size() != capacity());
        _data[_next.fetch_add(1, std::memory_order_acquire)] = value;
    }
    void insert(value_type&& value) noexcept
    {
        assert(size() != capacity());
        _data[_next.fetch_add(1, std::memory_order_acquire)] = std::move(value);
    }
public:
    template<
        typename ElementTypeO,
        typename ElementAllocatorO,
        typename FixedStorageTypeO
            >
    friend std::ostream& operator<<(
        std::ostream& o,
        const lockfree_fixed_size_bucket_t<ElementTypeO, ElementAllocatorO, FixedStorageTypeO>& bucket
    );
};

template<
    typename ElementType,
    typename ElementAllocator = std::allocator<ElementType>,
    typename FixedStorageType = fixed_size_dynamic_array<ElementType, ElementAllocator>,
    typename BucketType = fixed_size_bucket_t<ElementType, ElementAllocator, FixedStorageType>,
    typename LockType = std::mutex
    >
class thread_safe_bucket_t
{
public:
    using bucket_type = BucketType;
    using element_allocator_t = typename bucket_type::element_allocator_t;
    using storage_type = typename bucket_type::storage_type;
    using lock_type = LockType;
public:
    using value_type = ElementType;
    using reference = value_type&;
    using const_reference = const value_type&;

    using iterator = typename bucket_type::iterator;
    using const_iterator = typename bucket_type::const_iterator;

    using size_type = typename bucket_type::size_type;
private:
    bucket_type _bucket = {};
    lock_type _lock = {};
public:
    constexpr thread_safe_bucket_t() noexcept = default;
    thread_safe_bucket_t(size_type initial_capacity, element_allocator_t alloc)
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
    [[nodiscard]] size_type size() const { return std::size(_bucket); }
public:
    constexpr iterator begin() noexcept { return std::begin(_bucket); }
    constexpr const_iterator begin() const noexcept { return std::begin(_bucket); }
    constexpr iterator end() noexcept { return std::end(_bucket); }
    constexpr const_iterator end() const noexcept { return std::end(_bucket); }
public:
    lock_type& lock() noexcept { return _lock; }
    const lock_type& lock() const noexcept { return _lock; }
public:
    constexpr reference at(size_type index) noexcept { return _bucket[index]; }
    constexpr const_reference at(size_type index) const noexcept { return _bucket[index]; }

    constexpr reference operator[](size_type index) noexcept { return at(index); }
    constexpr const_reference operator[](size_type index) const noexcept { return at(index); }
public:
    void insert(const_reference value) noexcept
    {
        std::lock_guard{_lock};
        _bucket.insert(value);
    }
    void insert(value_type&& value) noexcept
    {
        std::lock_guard{_lock};
        _bucket.insert(std::move(value));
    }
public:
    template<
        typename ElementTypeO,
        typename ElementAllocatorO,
        typename FixedStorageTypeO,
        typename BucketTypeO,
        typename LockTypeO
        >
    friend std::ostream& operator<<(
        std::ostream& o,
        const thread_safe_bucket_t<ElementTypeO, ElementAllocatorO, FixedStorageTypeO, BucketTypeO, LockTypeO>& bucket
        );
};

template<
    typename ElementType,
    typename ElementAllocator,
    typename FixedStorageType,
    typename BucketType,
    typename LockType
        >
std::ostream& operator<<(
    std::ostream& o,
    const thread_safe_bucket_t<ElementType, ElementAllocator, FixedStorageType, BucketType, LockType>& bucket
)
{
    o << "variable(" << bucket.size() << ')';
    o << '[';
    for(const auto& v : bucket) o << v << ", ";
    o << ']';
    return o;
}

template<
    typename ElementType,
    typename ElementAllocator,
    typename FixedStorageType
    >
std::ostream& operator<<(
    std::ostream& o,
    const lockfree_fixed_size_bucket_t<ElementType, ElementAllocator, FixedStorageType>& bucket
)
{
    o << "variable(" << bucket.size() << ')';
    o << '[';
    for(const auto& v : bucket) o << v << ", ";
    o << ']';
    return o;
}
}
