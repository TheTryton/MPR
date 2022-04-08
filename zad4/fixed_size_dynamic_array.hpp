#pragma once

#include <memory>
#include <ostream>

#include <assert.h>

template<typename T, typename AllocT = std::allocator<T>>
class fixed_size_dynamic_array
{
private:
    struct deleter_t
    {
        AllocT alloc = {};
        size_t size = {};
        void operator()(T* ptr)
        {
            if(ptr == nullptr)
                return;
            std::destroy_n(ptr, size);
            std::allocator_traits<AllocT>::deallocate(alloc, ptr, size);
        }
    };
private:
    std::unique_ptr<T[], deleter_t> _data = nullptr;
    size_t _size = 0;
private:
    constexpr static std::unique_ptr<T[], deleter_t> prepare_data(size_t size, AllocT alloc, const T& init)
    {
        if(size == 0)
            return nullptr;

        auto memory = std::allocator_traits<AllocT>::allocate(alloc, size);
        std::uninitialized_fill_n(memory, size, init);

        return std::unique_ptr<T[], deleter_t>(memory, deleter_t{ .alloc=alloc, .size = size });
    }
    constexpr static std::unique_ptr<T[], deleter_t> prepare_data(size_t size, AllocT alloc)
    {
        if(size == 0)
            return nullptr;

        auto memory = std::allocator_traits<AllocT>::allocate(alloc, size);
        if constexpr(!std::is_trivially_default_constructible_v<T> && std::is_default_constructible_v<T>)
            std::uninitialized_fill_n(memory, size, T{});

        return std::unique_ptr<T[], deleter_t>(memory, deleter_t{ .alloc=alloc, .size = size });
    }
    constexpr static std::unique_ptr<T[], deleter_t> prepare_data(size_t size, AllocT alloc, const T* init_b, const T* init_e)
    {
        if(size == 0)
            return nullptr;

        auto memory = std::allocator_traits<AllocT>::allocate(alloc, size);
        std::uninitialized_copy_n(init_b, size, memory);

        return std::unique_ptr<T[], deleter_t>(memory, deleter_t{ .alloc=alloc, .size = size });
    }
public:
    constexpr fixed_size_dynamic_array() noexcept = default;
    explicit fixed_size_dynamic_array(size_t size, AllocT alloc = {})
        : _data(prepare_data(size, alloc))
        , _size(size)
    {}

    fixed_size_dynamic_array(size_t size, const T& init, AllocT alloc = {})
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
    [[nodiscard]] constexpr size_t size() const noexcept { return _size; }

    constexpr T* data() noexcept { return _data.get(); }
    constexpr const T* data() const noexcept { return _data.get(); }

    constexpr T* begin() noexcept { return data(); }
    constexpr const T* begin() const noexcept { return data(); }

    constexpr T* end() noexcept { return data() + size(); }
    constexpr const T* end() const noexcept { return data() + size(); }
public:
    constexpr T& at(size_t index) noexcept { assert(index < _size); return _data[index]; }
    constexpr const T& at(size_t index) const noexcept { assert(index < _size); return _data[index]; }

    constexpr T& operator[](size_t index) noexcept { return at(index); }
    constexpr const T& operator[](size_t index) const noexcept { return at(index); }
public:
    template<typename TO, typename AllocTO>
    friend std::ostream& operator<<(std::ostream& o, const fixed_size_dynamic_array<TO, AllocTO>& farr);
};

template<typename T, typename AllocT>
std::ostream& operator<<(std::ostream& o, const fixed_size_dynamic_array<T, AllocT>& farr)
{
    o << '[';
    for(const auto& v : farr) o << v << ", ";
    o << ']' << '\n';
    return o;
}
