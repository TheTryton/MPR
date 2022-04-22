#pragma once

#include <memory>
#include <ostream>
#include <iostream>

#include <assert.h>

template<
    typename ElementType,
    typename ElementAllocator = std::allocator<ElementType>
    >
class fixed_size_dynamic_array
{
public:
    using element_allocator_t = ElementAllocator;
public:
    using value_type = ElementType;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;

    using iterator = pointer;
    using const_iterator = const_pointer;

    using size_type = std::size_t;
private:
    struct deleter_t
    {
        ElementAllocator alloc = {};
        size_type size = {};
        void operator()(pointer ptr)
        {
            if(ptr == nullptr)
                return;
            std::default_delete<value_type>();
            std::cout << "Deallocating " << ptr << std::endl;
            std::destroy_n(ptr, size);
            std::cout << "Deallocated " << ptr << std::endl;
            std::allocator_traits<element_allocator_t>::deallocate(alloc, ptr, size);
        }
    };
private:
    std::unique_ptr<value_type[], deleter_t> _data = nullptr;
    size_type _size = 0;
private:
    constexpr static std::unique_ptr<value_type[], deleter_t> prepare_data(size_type size, element_allocator_t alloc, const_reference init)
    {
        if(size == 0)
            return nullptr;

        auto memory = std::allocator_traits<element_allocator_t>::allocate(alloc, size);
        std::cout << "Size=" << size << std::endl;
        std::uninitialized_fill_n(memory, size, init);

        return std::unique_ptr<value_type[], deleter_t>(memory, deleter_t{ .alloc=alloc, .size = size });
    }
    constexpr static std::unique_ptr<value_type[], deleter_t> prepare_data(size_type size, element_allocator_t alloc)
    {
        if(size == 0)
            return nullptr;

        auto memory = std::allocator_traits<element_allocator_t>::allocate(alloc, size);
        std::cout << "Size=" << size << std::endl;
        if constexpr(!std::is_trivially_default_constructible_v<value_type> && std::is_default_constructible_v<value_type>)
            std::uninitialized_fill_n(memory, size, value_type{});

        return std::unique_ptr<value_type[], deleter_t>(memory, deleter_t{ .alloc=alloc, .size = size });
    }
    template<typename It>
    constexpr static std::unique_ptr<value_type[], deleter_t> prepare_data(size_type size, element_allocator_t alloc, It init_b, It init_e)
    {
        if(size == 0)
            return nullptr;

        std::cout << "Size==init_e - init_b? " << (std::distance(init_b, init_e) == size) << std::endl;
        auto memory = std::allocator_traits<element_allocator_t>::allocate(alloc, size);
        std::uninitialized_copy(init_b, init_e, memory);

        return std::unique_ptr<value_type[], deleter_t>(memory, deleter_t{ .alloc=alloc, .size = size });
    }
public:
    constexpr fixed_size_dynamic_array() noexcept = default;
    explicit fixed_size_dynamic_array(size_type size, element_allocator_t alloc = {})
        : _data(prepare_data(size, alloc))
        , _size(size)
    {}

    fixed_size_dynamic_array(size_type size, const_reference init, element_allocator_t alloc = {})
        : _data(prepare_data(size, alloc, init))
        , _size(size)
    {}
    fixed_size_dynamic_array(const fixed_size_dynamic_array& other)
        : _data(prepare_data(other._size, other._data.get_deleter().alloc, std::begin(other), std::end(other)))
        , _size(other._size)
    { }
    fixed_size_dynamic_array(fixed_size_dynamic_array&& other) noexcept
        : _data(std::move(other._data))
        , _size(std::move(other._size))
    {
        other._size = 0;
    }
public:
    fixed_size_dynamic_array& operator=(const fixed_size_dynamic_array& other)
    {
        _data = prepare_data(other._size, other._data.get_deleter().alloc, std::begin(other), std::end(other));
        _size = other._size;
        return *this;
    }
    fixed_size_dynamic_array& operator=(fixed_size_dynamic_array&& other) noexcept
    {
        _data = std::move(other._data);
        _size = std::move(other._size);
        other._size = 0;
        return *this;
    }
public:
    [[nodiscard]] constexpr size_type size() const noexcept { return _size; }

    constexpr pointer data() noexcept { return _data.get(); }
    constexpr const_pointer data() const noexcept { return _data.get(); }

    constexpr pointer begin() noexcept { return data(); }
    constexpr const_pointer begin() const noexcept { return data(); }

    constexpr pointer end() noexcept { return data() + size(); }
    constexpr const_pointer end() const noexcept { return data() + size(); }
public:
    constexpr reference at(size_type index) noexcept { assert(index < _size); return _data[index]; }
    constexpr const_reference at(size_type index) const noexcept { assert(index < _size); return _data[index]; }

    constexpr reference operator[](size_type index) noexcept { return at(index); }
    constexpr const_reference operator[](size_type index) const noexcept { return at(index); }
public:
    template<
        typename ElementTypeO,
        typename ElementAllocatorO
        >
    friend std::ostream& operator<<(
        std::ostream& o,
        const fixed_size_dynamic_array<ElementTypeO, ElementAllocatorO>& farr
        );
};

template<
    typename ElementType,
    typename ElementAllocator
    >
std::ostream& operator<<(
    std::ostream& o,
    const fixed_size_dynamic_array<ElementType, ElementAllocator>& farr
    )
{
    o << '[';
    for(const auto& v : farr) o << v << ", ";
    o << ']' << '\n';
    return o;
}
