#pragma once

#include <bucket.hpp>
#include <lockables.hpp>

#include <value_range.hpp>

template<
    typename ElementType,
    typename ElementAllocator = std::allocator<ElementType>,
    typename BucketType = fixed_size_bucket_t<ElementType, ElementAllocator>,
    typename BucketAllocator = std::allocator<BucketType>
    >
class buckets_t
{
public:
    using element_type = ElementType;
    using element_allocator_t = ElementAllocator;
    using bucket_type = BucketType;
    using bucket_allocator_t = BucketAllocator;
public:
    using value_type = bucket_type;
    using reference = value_type&;
    using const_reference = const value_type&;

    using iterator = typename fixed_size_dynamic_array<bucket_type, bucket_allocator_t>::iterator;
    using const_iterator = typename fixed_size_dynamic_array<bucket_type, bucket_allocator_t>::const_iterator;

    using size_type = typename fixed_size_dynamic_array<bucket_type, bucket_allocator_t>::size_type;
private:
    fixed_size_dynamic_array<bucket_type, bucket_allocator_t> _buckets = {};
    value_range<element_type> _values_range = {};
public:
    constexpr buckets_t() noexcept = default;
    buckets_t(
        size_type bucket_count, size_type bucket_size, const value_range<element_type>& inserted_values_range,
        element_allocator_t element_allocator = {}, bucket_allocator_t bucket_allocator = {}
    )
        : _buckets(bucket_count, bucket_type{bucket_size, std::move(element_allocator)}, std::move(bucket_allocator))
        , _values_range(inserted_values_range)
    {}
    buckets_t(const buckets_t& other) = default;
    buckets_t(buckets_t&& other) noexcept = default;
public:
    buckets_t& operator=(const buckets_t& other) = default;
    buckets_t& operator=(buckets_t&& other) noexcept = default;
public:
    [[nodiscard]] constexpr size_type size() const noexcept { return std::size(_buckets); }
    [[nodiscard]] constexpr size_type total_size() const noexcept
    {
        return std::accumulate(std::begin(_buckets), std::end(_buckets), size_type(0), [](auto a, auto b){ return a + std::size(b);});
    }
public:
    constexpr iterator begin()  noexcept { return std::begin(_buckets); }
    constexpr const_iterator begin() const noexcept { return std::begin(_buckets); }
    constexpr iterator end() noexcept { return std::end(_buckets); }
    constexpr const_iterator end() const noexcept { return std::end(_buckets); }
public:
    constexpr reference at(size_type bucket_index) noexcept { return _buckets[bucket_index]; }
    constexpr const_reference at(size_type bucket_index) const noexcept { return _buckets[bucket_index]; }

    constexpr reference operator[](size_type bucket_index) noexcept { return at(bucket_index); }
    constexpr const_reference operator[](size_type bucket_index) const noexcept { return at(bucket_index); }
private:
    constexpr size_type select_bucket(const element_type& value) const noexcept
    {
        assert(can_accept_value(value));
        auto coefficent01 = (value - _values_range.low) / _values_range.length();
        return static_cast<size_t>(std::round(coefficent01 * (size() - 1)));
    }
public:
    constexpr bool can_accept_value(const element_type& value) const noexcept
    {
        return _values_range.contains(value);
    }
    constexpr value_range<element_type> bucket_value_range(size_type bucket_index) const noexcept
    {
        return _values_range.split_index(size(), bucket_index);
    }
public:
    void insert(const element_type& value) noexcept
    {
        _buckets[select_bucket(value)].insert(value);
    }
    void insert(element_type&& value) noexcept
    {
        _buckets[select_bucket(value)].insert(std::move(value));
    }
public:
    template<
        typename ElementTypeO,
        typename ElementAllocatorO,
        typename BucketTypeO,
        typename BucketAllocatorO
        >
    friend std::ostream& operator<<(
        std::ostream& o,
        const buckets_t<
            ElementTypeO, ElementAllocatorO,
            BucketTypeO, BucketAllocatorO>& buckets
        );
};

template<
    typename ElementType,
    typename ElementAllocator,
    typename BucketType,
    typename BucketAllocator
    >
std::ostream& operator<<(
    std::ostream& o,
    const buckets_t<ElementType, ElementAllocator, BucketType, BucketAllocator>& buckets
    )
{
    size_t i = 0;
    for(const auto& bucket : buckets)
    {
        o << '{' << i++ << "} -> " << bucket << '\n';
    }
    return o;
}

template<typename BucketType, size_t BucketAlignment>
class alignas(BucketAlignment) aligned_bucket_t
    : public BucketType
{
    using BucketType::BucketType;
};

template<
    typename ElementType,
    typename ElementAllocator = std::allocator<ElementType>,
    typename BaseBucketType = fixed_size_bucket_t<ElementType, ElementAllocator>,
    size_t BucketAlignmentOverride = std::alignment_of_v<BaseBucketType>,
    typename AlignedBucketAllocator = std::allocator<aligned_bucket_t<BaseBucketType, BucketAlignmentOverride>>,
    typename LockType = std::mutex
    >
class thread_safe_buckets_t
{
public:
    using element_type = ElementType;
    using element_allocator_t = ElementAllocator;
    using bucket_type = aligned_bucket_t<BaseBucketType, BucketAlignmentOverride>;
    using bucket_allocator_t = AlignedBucketAllocator;
    using lock_type = LockType;
public:
    using value_type = bucket_type;
    using reference = value_type&;
    using const_reference = const value_type&;

    using iterator = typename bucket_type::iterator;
    using const_iterator = typename bucket_type::const_iterator;

    using size_type = typename bucket_type::size_type;
private:
    buckets_t<element_type, element_allocator_t, bucket_type, bucket_allocator_t> _buckets = {};
    lock_type _lock = {};
public:
    constexpr thread_safe_buckets_t() noexcept = default;
    thread_safe_buckets_t(
        size_type bucket_count, size_type bucket_size, const value_range<element_type>& inserted_values_range,
        element_allocator_t alloc = {}, bucket_allocator_t bt_alloc = {}
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
    [[nodiscard]] constexpr size_type size() const noexcept { return std::size(_buckets); }
    [[nodiscard]] constexpr size_type total_size() const noexcept
    {
        return std::accumulate(std::begin(_buckets), std::end(_buckets), size_type(0), [](auto a, auto b){ return a + std::size(b);});
    }
public:
    constexpr iterator begin()  noexcept { return std::begin(_buckets); }
    constexpr const_iterator begin() const noexcept { return std::begin(_buckets); }
    constexpr iterator end() noexcept { return std::end(_buckets); }
    constexpr const_iterator end() const noexcept { return std::end(_buckets); }
public:
    lock_type& lock() noexcept { return _lock; }
    const lock_type& lock() const noexcept { return _lock; }
public:
    constexpr reference at(size_type bucket_index) noexcept { return _buckets[bucket_index]; }
    constexpr const_reference at(size_type bucket_index) const noexcept { return _buckets[bucket_index]; }

    constexpr reference operator[](size_type bucket_index) noexcept { return at(bucket_index); }
    constexpr const_reference operator[](size_type bucket_index) const noexcept { return at(bucket_index); }
public:
    constexpr bool can_accept_value(const element_type& value) const noexcept
    {
        return _buckets.can_accept_value(value);
    }
    constexpr value_range<element_type> bucket_value_range(size_type bucket_index) const noexcept
    {
        return _buckets.bucket_value_range(bucket_index);
    }
public:
    void insert(const element_type& value) noexcept
    {
        std::lock_guard{_lock};
        _buckets[select_bucket(value)].insert(value);
    }
    void insert(element_type&& value) noexcept
    {
        std::lock_guard{_lock};
        _buckets[select_bucket(value)].insert(std::move(value));
    }
public:
    template<
        typename ElementTypeO,
        typename ElementAllocatorO,
        typename BaseBucketTypeO,
        size_t BucketAlignmentOverrideO,
        typename AlignedBucketAllocatorO,
        typename LockTypeO
            >
    friend std::ostream& operator<<(
        std::ostream& o,
        const thread_safe_buckets_t<
            ElementTypeO, ElementAllocatorO,
            BaseBucketTypeO, BucketAlignmentOverrideO,
            AlignedBucketAllocatorO, LockTypeO>& buckets
    );
};

template<
    typename ElementType,
    typename ElementAllocator,
    typename BaseBucketType,
    size_t BucketAlignmentOverride,
    typename AlignedBucketAllocator,
    typename LockType
    >
std::ostream& operator<<(
    std::ostream& o,
    const thread_safe_buckets_t<
        ElementType, ElementAllocator,
        BaseBucketType, BucketAlignmentOverride,
        AlignedBucketAllocator, LockType>& buckets
    )
{
    size_t i = 0;
    for(const auto& bucket : buckets)
    {
        o << '{' << i++ << "} -> " << bucket << '\n';
    }
    return o;
}
