#pragma once

#include <fixed_size_dynamic_array.hpp>

#include <assert.h>
#include <vector>
#include <mutex>
#include <atomic>

template<typename Alloc, typename T>
concept Allocator = requires(Alloc& alloc, typename std::allocator_traits<Alloc>::pointer p, typename std::allocator_traits<Alloc>::size_type n)
{
    requires std::same_as<typename Alloc::value_type, T>;
    { alloc.allocate(n) } -> std::same_as<typename std::allocator_traits<Alloc>::pointer>;
    { alloc.deallocate(p, n) };
};

template<typename FixedStorageType, typename ElementType, typename ElementAllocator>
concept FixedSizeStorage = requires(FixedStorageType storage, ElementType element, ElementAllocator allocator, typename FixedStorageType::size_type capacity)
{
    typename FixedStorageType::size_type;

    requires std::is_default_constructible_v<FixedStorageType>;
    requires std::is_constructible_v<FixedStorageType, typename FixedStorageType::size_type, ElementAllocator>;

    requires std::is_copy_constructible_v<FixedStorageType>;
    requires std::is_move_constructible_v<FixedStorageType>;

    requires std::is_copy_assignable_v<FixedStorageType>;
    requires std::is_move_assignable_v<FixedStorageType>;

    { storage.begin() } -> std::random_access_iterator;
    { storage.end() } -> std::random_access_iterator;
    { storage[capacity] } -> std::convertible_to<ElementType>;
    { storage[capacity] = element } -> std::convertible_to<ElementType>;
    { storage.at(capacity) } -> std::convertible_to<ElementType>;
    { storage.at(capacity) = element } -> std::convertible_to<ElementType>;
};

template<typename VariableStorageType, typename ElementType, typename ElementAllocator>
concept VariableSizeStorage = requires(VariableStorageType storage, ElementType element, ElementAllocator allocator, typename VariableStorageType::size_type capacity)
{
    typename VariableStorageType::size_type;

    requires std::is_default_constructible_v<VariableStorageType>;
    requires std::is_constructible_v<VariableStorageType, ElementAllocator>;

    requires std::is_copy_constructible_v<VariableStorageType>;
    requires std::is_move_constructible_v<VariableStorageType>;

    requires std::is_copy_assignable_v<VariableStorageType>;
    requires std::is_move_assignable_v<VariableStorageType>;

    { storage.begin() } -> std::random_access_iterator;
    { storage.end() } -> std::random_access_iterator;
    { storage[capacity] } -> std::convertible_to<ElementType>;
    { storage[capacity] = element } -> std::convertible_to<ElementType>;
    { storage.at(capacity) } -> std::convertible_to<ElementType>;
    { storage.at(capacity) = element } -> std::convertible_to<ElementType>;

    { storage.push_back(element) };
};

template<class VariableStorageType>
concept HasReserveMethod = requires(VariableStorageType storage, typename VariableStorageType::size_type capacity)
{
    {storage.reserve(capacity) };
};

template<class LockType>
concept BasicLockable = requires(LockType lock)
{
    { lock.lock() };
    { lock.unlock() };
};

template<class LockType>
concept Lockable = BasicLockable<LockType> && requires(LockType lock)
{
    { lock.try_lock() } -> std::convertible_to<bool>;
};

template<typename BucketType, typename ElementType, typename ElementAllocator>
concept Bucket = requires(BucketType bucket, ElementType element, ElementAllocator allocator, typename BucketType::size_type capacity)
{
    typename BucketType::size_type;

    requires std::is_default_constructible_v<BucketType>;
    requires std::is_constructible_v<BucketType, typename BucketType::size_type, ElementAllocator>;

    requires std::is_copy_constructible_v<BucketType>;
    requires std::is_move_constructible_v<BucketType>;

    requires std::is_copy_assignable_v<BucketType>;
    requires std::is_move_assignable_v<BucketType>;

    { bucket.begin() } -> std::random_access_iterator;
    { bucket.end() } -> std::random_access_iterator;
    { bucket[capacity] } -> std::convertible_to<ElementType>;
    { bucket.at(capacity) } -> std::convertible_to<ElementType>;

    { bucket.size() } -> std::same_as<typename BucketType::size_type>;

    { bucket.insert(element) };
    { bucket.insert(std::move(element)) };
};

template<
    typename ElementType,
    Allocator<ElementType> ElementAllocator = std::allocator<ElementType>,
    FixedSizeStorage<ElementType, ElementAllocator> FixedStorageType = fixed_size_dynamic_array<ElementType, ElementAllocator>
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
        Allocator<ElementTypeO> ElementAllocatorO,
        FixedSizeStorage<ElementTypeO, ElementAllocatorO> FixedStorageTypeO
        >
    friend std::ostream& operator<<(
        std::ostream& o,
        const fixed_size_bucket_t<ElementTypeO, ElementAllocatorO, FixedStorageTypeO>& bucket
        );
};

static_assert(
    Bucket<
    fixed_size_bucket_t<
        int,
        std::allocator<int>,
        fixed_size_dynamic_array<int, std::allocator<int>>
    >,
    int,
    std::allocator<int>
>);

template<
    typename ElementType,
    Allocator<ElementType> ElementAllocator = std::allocator<ElementType>,
    VariableSizeStorage<ElementType, ElementAllocator> VariableStorageType = std::vector<ElementType, ElementAllocator>
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
        if constexpr (HasReserveMethod<storage_type>)
        {
            _data.reserve(initial_capacity);
        }
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
        Allocator<ElementTypeO> ElementAllocatorO,
        VariableSizeStorage<ElementTypeO, ElementAllocatorO> VariableStorageTypeO
        >
    friend std::ostream& operator<<(
        std::ostream& o,
        const variable_size_bucket_t<ElementTypeO, ElementAllocatorO, VariableStorageTypeO>& bucket
        );
};

static_assert(
    Bucket<
    variable_size_bucket_t<
        int,
        std::allocator<int>,
        std::vector<int, std::allocator<int>>
    >,
    int,
    std::allocator<int>
>);

template<
    typename ElementType,
    Allocator<ElementType> ElementAllocator,
    FixedSizeStorage<ElementType, ElementAllocator> FixedStorageType
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
    Allocator<ElementType> ElementAllocator,
    VariableSizeStorage<ElementType, ElementAllocator> VariableStorageType
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
    Allocator<ElementType> ElementAllocator = std::allocator<ElementType>,
    FixedSizeStorage<ElementType, ElementAllocator> FixedStorageType = fixed_size_dynamic_array<ElementType, ElementAllocator>
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
    lockfree_fixed_size_bucket_t(const lockfree_fixed_size_bucket_t& other)
        : _data(other._data)
        , _next(other._next.load())
    {}
    lockfree_fixed_size_bucket_t(lockfree_fixed_size_bucket_t&& other) noexcept
        : _data(std::move(other._data))
        , _next(std::move(other._next))
    {}
public:
    lockfree_fixed_size_bucket_t& operator=(const lockfree_fixed_size_bucket_t& other)
    {
        _data = other._data;
        _next = other._next.load();
        return *this;
    }
    lockfree_fixed_size_bucket_t& operator=(lockfree_fixed_size_bucket_t&& other) noexcept
    {
        _data = std::move(other._data);
        _next = std::move(other._next);
        return *this;
    }
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
        Allocator<ElementTypeO> ElementAllocatorO,
        FixedSizeStorage<ElementTypeO, ElementAllocatorO> FixedStorageTypeO
    >
    friend std::ostream& operator<<(
        std::ostream& o,
        const lockfree_fixed_size_bucket_t<ElementTypeO, ElementAllocatorO, FixedStorageTypeO>& bucket
    );
};

static_assert(
    Bucket<
    lockfree_fixed_size_bucket_t<
        int,
        std::allocator<int>,
        fixed_size_dynamic_array<int, std::allocator<int>>
    >,
    int,
    std::allocator<int>
>);

template<
    typename ElementType,
    Allocator<ElementType> ElementAllocator = std::allocator<ElementType>,
    Bucket<ElementType, ElementAllocator> BucketType = fixed_size_bucket_t<ElementType, ElementAllocator>,
    Lockable LockType = std::mutex
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
        : _bucket(other._bucket)
    { }
    thread_safe_bucket_t(thread_safe_bucket_t&& other)
        : _bucket(std::move(other._bucket))
    { }
public:
    thread_safe_bucket_t& operator=(const thread_safe_bucket_t& other)
    {
        _bucket = other._bucket;
        return *this;
    }
    thread_safe_bucket_t& operator=(thread_safe_bucket_t&& other)
    {
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
        std::lock_guard l{_lock};
        _bucket.insert(value);
    }
    void insert(value_type&& value) noexcept
    {
        std::lock_guard l{_lock};
        _bucket.insert(std::move(value));
    }
public:
    template<
        typename ElementTypeO,
        Allocator<ElementTypeO> ElementAllocatorO,
        Bucket<ElementTypeO, ElementAllocatorO> BucketTypeO,
        Lockable LockTypeO
        >
    friend std::ostream& operator<<(
        std::ostream& o,
        const thread_safe_bucket_t<ElementTypeO, ElementAllocatorO, BucketTypeO, LockTypeO>& bucket
        );
};

static_assert(
    Bucket<
    thread_safe_bucket_t<
        int,
        std::allocator<int>,
        fixed_size_bucket_t<int, std::allocator<int>>
    >,
    int,
    std::allocator<int>
>);

static_assert(
    Bucket<
    thread_safe_bucket_t<
        int,
        std::allocator<int>,
        variable_size_bucket_t<int, std::allocator<int>>
    >,
    int,
    std::allocator<int>
>);

template<
    typename ElementType,
    Allocator<ElementType> ElementAllocator,
    FixedSizeStorage<ElementType, ElementAllocator> FixedStorageType
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

template<
    typename ElementType,
    Allocator<ElementType>  ElementAllocator,
    Bucket<ElementType, ElementAllocator> BucketType,
    Lockable LockType
>
std::ostream& operator<<(
    std::ostream& o,
    const thread_safe_bucket_t<ElementType, ElementAllocator, BucketType, LockType>& bucket
)
{
    o << "variable(" << bucket.size() << ')';
    o << '[';
    for(const auto& v : bucket) o << v << ", ";
    o << ']';
    return o;
}
}
