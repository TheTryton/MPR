#pragma once

#include <bucket.hpp>
#include <lockables.hpp>

#include <value_range.hpp>

template<typename KeyType>
concept BucketsKey =
#ifdef _MSC_VER
true;
#else
RangeBoundary<KeyType>&&
requires(KeyType key)
{
    { std::round(key) } -> std::convertible_to<KeyType>;
    { key } -> std::convertible_to<size_t>;
};
#endif

template<typename BucketsType, typename ElementType, typename KeyFunc, typename ElementAllocator, typename BucketType, typename BucketAllocator>
concept Buckets =
#ifdef _MSC_VER
true;
#else
requires(BucketsType buckets, ElementType element, typename BucketType::size_type n)
{
    typename BucketsType::size_type;

    requires std::is_default_constructible_v<BucketsType>;
    requires std::is_constructible_v<BucketsType,
        typename BucketsType::size_type,
        typename BucketsType::size_type,
        value_range<ElementType>,
        KeyFunc,
        ElementAllocator,
        BucketAllocator>;

    requires std::is_copy_constructible_v<BucketsType>;
    requires std::is_move_constructible_v<BucketsType>;

    requires std::is_copy_assignable_v<BucketsType>;
    requires std::is_move_assignable_v<BucketsType>;

    { buckets.begin() } -> std::random_access_iterator;
    { buckets.end() } -> std::random_access_iterator;

    requires Bucket<typename std::iterator_traits<decltype(buckets.begin())>::value_type, ElementType, ElementAllocator>;
    requires Bucket<typename std::iterator_traits<decltype(buckets.end())>::value_type, ElementType, ElementAllocator>;

    { buckets[n] } -> std::convertible_to<BucketType>;
    { buckets.at(n) } -> std::convertible_to<BucketType>;

    { buckets.size() } -> std::same_as<typename BucketType::size_type>;
    { buckets.total_size() } -> std::same_as<typename BucketType::size_type>;

    { buckets.can_accept_value(element) } -> std::convertible_to<bool>;
    { buckets.bucket_value_range(n) } -> std::same_as<value_range<ElementType>>;

    { buckets.insert(element) };
    { buckets.insert(std::move(element)) };
};
#endif

template<typename ElementType = void>
struct identity_key
{
    constexpr ElementType operator()(const ElementType& v) const noexcept
    {
        return v;
    }
};

template<>
struct identity_key<void>
{
    template<typename ElementType>
    constexpr ElementType operator()(const ElementType& v) const noexcept
    {
        return v;
    }
};

template<typename ElementType = void>
struct reverse_key
{
    constexpr ElementType operator()(const ElementType& v) const noexcept
    {
        return -v;
    }
};

template<>
struct reverse_key<void>
{
    template<typename ElementType>
    constexpr ElementType operator()(const ElementType& v) const noexcept
    {
        return -v;
    }
};

#ifdef _MSC_VER
template<
    typename ElementType,
    typename KeyFunc = identity_key<ElementType>,
    typename ElementAllocator = std::allocator<ElementType>,
    typename BucketType = fixed_size_bucket_t<ElementType, ElementAllocator>,
    typename BucketAllocator = std::allocator<BucketType>
>
#else
template<
    typename ElementType,
    std::invocable<const ElementType&> KeyFunc = identity_key<ElementType>,
    Allocator<ElementType> ElementAllocator = std::allocator<ElementType>,
    Bucket<ElementType, ElementAllocator> BucketType = fixed_size_bucket_t<ElementType, ElementAllocator>,
    Allocator<BucketType> BucketAllocator = std::allocator<BucketType>
    >
requires BucketsKey<std::invoke_result_t<KeyFunc, const ElementType&>>
#endif
class buckets_t
{
public:
    using element_type = ElementType;
    using element_allocator_t = ElementAllocator;
    using bucket_type = BucketType;
    using bucket_allocator_t = BucketAllocator;
public:
    using value_type = bucket_type;
    using key_type = std::invoke_result_t<KeyFunc, const ElementType&>;
    using reference = value_type&;
    using const_reference = const value_type&;

    using iterator = typename fixed_size_dynamic_array<bucket_type, bucket_allocator_t>::iterator;
    using const_iterator = typename fixed_size_dynamic_array<bucket_type, bucket_allocator_t>::const_iterator;

    using size_type = typename fixed_size_dynamic_array<bucket_type, bucket_allocator_t>::size_type;
private:
    fixed_size_dynamic_array<bucket_type, bucket_allocator_t> _buckets = {};
    value_range<key_type> _values_range = {};
    KeyFunc _key_func = {};
public:
    constexpr buckets_t() noexcept = default;
    buckets_t(
        size_type bucket_count, size_type bucket_size,
        const value_range<key_type>& inserted_values_range,
        KeyFunc key_func = {},
        element_allocator_t element_allocator = {}, bucket_allocator_t bucket_allocator = {}
    )
        : _buckets(bucket_count, bucket_type{bucket_size, std::move(element_allocator)}, std::move(bucket_allocator))
        , _values_range(inserted_values_range)
        , _key_func(key_func)
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
        auto coefficent01 = (_key_func(value) - _values_range.low) / _values_range.length();
        return static_cast<size_t>(std::round(coefficent01 * (size() - size_type(1))));
    }
public:
    constexpr bool can_accept_value(const element_type& value) const noexcept
    {
        return _values_range.contains(_key_func(value));
    }
    constexpr value_range<element_type> bucket_value_range(size_type bucket_index) const noexcept
    {
        return _values_range.split_index(bucket_index, size());
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
        Allocator<ElementTypeO> ElementAllocatorO,
        Bucket<ElementTypeO, ElementAllocatorO> BucketTypeO,
        Allocator<BucketTypeO> BucketAllocatorO
        >
    friend std::ostream& operator<<(
        std::ostream& o,
        const buckets_t<
            ElementTypeO, ElementAllocatorO,
            BucketTypeO, BucketAllocatorO>& buckets
        );
};

static_assert(Buckets<
    buckets_t<
        int,
        identity_key<int>,
        std::allocator<int>,
        fixed_size_bucket_t<int, std::allocator<int>>,
        std::allocator<fixed_size_bucket_t<int, std::allocator<int>>>
    >,
    int,
    identity_key<int>,
    std::allocator<int>,
    fixed_size_bucket_t<int, std::allocator<int>>,
    std::allocator<fixed_size_bucket_t<int, std::allocator<int>>>
>);
static_assert(Buckets<
    buckets_t<
        int,
        identity_key<int>,
        std::allocator<int>,
        variable_size_bucket_t<int, std::allocator<int>>,
        std::allocator<variable_size_bucket_t<int, std::allocator<int>>>
    >,
    int,
    identity_key<int>,
    std::allocator<int>,
    variable_size_bucket_t<int, std::allocator<int>>,
    std::allocator<variable_size_bucket_t<int, std::allocator<int>>>
>);
static_assert(Buckets<
    buckets_t<
        int,
        identity_key<int>,
        std::allocator<int>,
        threadsafe::lockfree_fixed_size_bucket_t<int, std::allocator<int>>,
        std::allocator<threadsafe::lockfree_fixed_size_bucket_t<int, std::allocator<int>>>
    >,
    int,
    identity_key<int>,
    std::allocator<int>,
    threadsafe::lockfree_fixed_size_bucket_t<int, std::allocator<int>>,
    std::allocator<threadsafe::lockfree_fixed_size_bucket_t<int, std::allocator<int>>>
>);
static_assert(Buckets<
    buckets_t<
        int,
        identity_key<int>,
        std::allocator<int>,
        threadsafe::thread_safe_bucket_t<int, std::allocator<int>, fixed_size_bucket_t<int, std::allocator<int>>>,
        std::allocator<threadsafe::thread_safe_bucket_t<int, std::allocator<int>, fixed_size_bucket_t<int, std::allocator<int>>>>
    >,
    int,
    identity_key<int>,
    std::allocator<int>,
    threadsafe::thread_safe_bucket_t<int, std::allocator<int>, fixed_size_bucket_t<int, std::allocator<int>>>,
    std::allocator<threadsafe::thread_safe_bucket_t<int, std::allocator<int>, fixed_size_bucket_t<int, std::allocator<int>>>>
>);
static_assert(Buckets<
    buckets_t<
        int,
        identity_key<int>,
        std::allocator<int>,
        threadsafe::thread_safe_bucket_t<int, std::allocator<int>, variable_size_bucket_t<int, std::allocator<int>>>,
        std::allocator<threadsafe::thread_safe_bucket_t<int, std::allocator<int>, variable_size_bucket_t<int, std::allocator<int>>>>
    >,
    int,
    identity_key<int>,
    std::allocator<int>,
    threadsafe::thread_safe_bucket_t<int, std::allocator<int>, variable_size_bucket_t<int, std::allocator<int>>>,
    std::allocator<threadsafe::thread_safe_bucket_t<int, std::allocator<int>, variable_size_bucket_t<int, std::allocator<int>>>>
>);

template<
    typename ElementType,
    Allocator<ElementType> ElementAllocator,
    Bucket<ElementType, ElementAllocator> BucketType,
    Allocator<BucketType> BucketAllocator
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
    Allocator<ElementType> ElementAllocator = std::allocator<ElementType>,
    Bucket<ElementType, ElementAllocator> BaseBucketType = fixed_size_bucket_t<ElementType, ElementAllocator>,
    size_t BucketAlignmentOverride = std::alignment_of_v<BaseBucketType>,
    Allocator<aligned_bucket_t<BaseBucketType, BucketAlignmentOverride>> AlignedBucketAllocator = std::allocator<aligned_bucket_t<BaseBucketType, BucketAlignmentOverride>>,
    Lockable LockType = std::mutex
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

    using iterator = typename buckets_t<element_type, element_allocator_t, bucket_type, bucket_allocator_t>::iterator;
    using const_iterator = typename buckets_t<element_type, element_allocator_t, bucket_type, bucket_allocator_t>::const_iterator;

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
    constexpr iterator begin() noexcept { return std::begin(_buckets); }
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
        std::lock_guard l{_lock};
        _buckets.insert(value);
    }
    void insert(element_type&& value) noexcept
    {
        std::lock_guard l{_lock};
        _buckets.insert(std::move(value));
    }
public:
    template<
        typename ElementTypeO,
        Allocator<ElementTypeO> ElementAllocatorO,
        Bucket<ElementTypeO, ElementAllocatorO> BaseBucketTypeO,
        size_t BucketAlignmentOverrideO,
        Allocator<aligned_bucket_t<BaseBucketTypeO, BucketAlignmentOverrideO>> AlignedBucketAllocatorO,
        Lockable LockTypeO
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
    Allocator<ElementType> ElementAllocator,
    Bucket<ElementType, ElementAllocator> BaseBucketType,
    size_t BucketAlignmentOverride,
    Allocator<aligned_bucket_t<BaseBucketType, BucketAlignmentOverride>> AlignedBucketAllocator,
    Lockable LockType
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
