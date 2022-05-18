#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <memory>
#include <memory_resource>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <string>
#include <string_view>
#include <array>

using namespace std;

template<typename T= char, typename Traits = char_traits<T>, typename BufferAllocator = std::allocator<T>>
class sequential_buffered_streambuf
    : public basic_streambuf<T, Traits>
{
private:
    struct deleter
    {
        BufferAllocator allocator;
        size_t size;

        constexpr void operator()(T* ptr) noexcept
        {
            if(ptr == nullptr)
                return;

            destroy_n(ptr, size);
            allocator_traits<BufferAllocator>::deallocate(allocator, ptr, size);
        }
    };
    std::basic_streambuf<T, Traits>* source;
    std::unique_ptr<T[], deleter> data;
private:
    static std::unique_ptr<T[], deleter> create_uninitialized(size_t buffer_size, const BufferAllocator& allocator)
    {
        auto alloc = allocator;
        auto memory = allocator_traits<BufferAllocator>::allocate(alloc, buffer_size);
        return {memory, deleter{.allocator = allocator, .size = buffer_size}};
    }
public:
    sequential_buffered_streambuf(streambuf* source, size_t buffer_size, const BufferAllocator& allocator)
        : source(source)
        , data(create_uninitialized(buffer_size, allocator))
    {}
    virtual ~sequential_buffered_streambuf() override = default;
protected:
    virtual streambuf::int_type underflow() override
    {
        auto count_read = source->sgetn(data.get(), data.get_deleter().size);
        if(count_read == 0)
            return Traits::eof();

        this->setg(data.get(), data.get(), data.get() + count_read);

        return Traits::to_int_type(*this->gptr());
    }

    virtual std::streamsize xsgetn(T* dest, std::streamsize count) override
    {
        std::streamsize read = 0;
        while(count > 0)
        {
            auto available = this->in_avail();
            if(available == 0)
                available = underflow();
            if(available == Traits::eof())
                return read;

            auto to_read = std::min(available, count);
            count -= to_read;
            read += to_read;

            dest = std::copy(this->gptr(), this->gptr() + to_read, dest);
            this->gbump(to_read);
        }

        return read;
    }
};

template<typename T = char, typename Traits = char_traits<T>, typename BufferAllocator = std::allocator<T>>
class sequential_buffered_stream
    : public basic_istream<T, Traits>
{
private:
    sequential_buffered_streambuf<T, Traits, BufferAllocator> stream_buf;
public:
    sequential_buffered_stream(istream& in, size_t buffer_size, const BufferAllocator& allocator)
        : basic_istream<T, Traits>(&stream_buf)
        , stream_buf(in.rdbuf(), buffer_size, allocator)
    {}
};

ptrdiff_t buffer_load(istream& in, char* to, size_t buffer_size) noexcept
{
    in.read(to, static_cast<std::streamsize>(buffer_size));
    return in.gcount();
}

template<typename F>
void foreach_chunked(istream& in, char* to, size_t buffer_size, F f) noexcept
{
    while(!in.eof())
    {
        auto read_n = buffer_load(in, to, buffer_size);
        std::for_each(to, to + read_n, f);
    }
}

template<typename T = char, typename Traits = char_traits<T>>
class istreambuf_buffered_iterator
{
public:
    using iterator_category = std::input_iterator_tag;
    using value_type = T;
    using difference_type = typename Traits::off_type;
    using pointer = T*;
    using reference = T;
    using char_type = T;
    using traits_type = Traits;
    using int_type = typename Traits::int_type;
    using streambuf_type = basic_streambuf<T, Traits>;
    using istream_type = basic_istream<T, Traits>;
private:
    streambuf_type* streambuf = nullptr;
    pointer buffer_begin = nullptr;
    pointer buffer_current = nullptr;
    pointer buffer_end = nullptr;
    pointer buffer_fill_end = nullptr;
private:
    void update_buffer()
    {
        auto available = streambuf->sgetn(buffer_begin, std::distance(buffer_begin, buffer_end));

        buffer_fill_end = buffer_begin + available;
        buffer_current = buffer_begin;
    }
    [[nodiscard]] constexpr bool reached_buffer_end() const noexcept
    {
        return buffer_current == buffer_fill_end;
    }
public:
    constexpr istreambuf_buffered_iterator() noexcept = default;
    istreambuf_buffered_iterator(streambuf_type* streambuf, pointer buffer_begin, pointer buffer_end)
        : streambuf(streambuf)
        , buffer_begin(buffer_begin)
        , buffer_current(buffer_begin)
        , buffer_end(buffer_end)
    {
        update_buffer();
    }
    constexpr istreambuf_buffered_iterator(const istreambuf_buffered_iterator& other) noexcept = default;
    constexpr istreambuf_buffered_iterator(istreambuf_buffered_iterator&& other) noexcept = default;
public:
    constexpr istreambuf_buffered_iterator& operator=(const istreambuf_buffered_iterator& other) noexcept = default;
    constexpr istreambuf_buffered_iterator& operator=(istreambuf_buffered_iterator&& other) noexcept = default;
public:
    constexpr reference operator*() const noexcept
    {
        return *buffer_current;
    }
    constexpr pointer operator->() const noexcept
    {
        return buffer_current;
    }
    constexpr istreambuf_buffered_iterator& operator++()
    {
        ++buffer_current;
        if(reached_buffer_end())
        {
            update_buffer();
        }

        return *this;
    }
public:
    constexpr bool operator==(const istreambuf_buffered_iterator& other) const noexcept
    {
        if(reached_buffer_end() && other.reached_buffer_end())
            return true;
        return false;
    }
    constexpr bool operator!=(const istreambuf_buffered_iterator& other) const noexcept
    {
        if(reached_buffer_end() && other.reached_buffer_end())
            return false;
        return true;
    }
};

constexpr size_t KBs(size_t c) noexcept
{
    return c * 1024;
}
constexpr size_t MBs(size_t c) noexcept
{
    return KBs(c) * 1024;
}
constexpr size_t GBs(size_t c) noexcept
{
    return MBs(c) * 1024;
}

constexpr size_t operator""_GB(unsigned long long c) noexcept
{
    return GBs(c);
}
constexpr size_t operator""_MB(unsigned long long c) noexcept
{
    return MBs(c);
}
constexpr size_t operator""_KB(unsigned long long c) noexcept
{
    return KBs(c);
}

size_t file_size(ifstream& file)
{
    file.seekg(ifstream::end);
    size_t size = file.tellg();
    file.seekg(ifstream::beg);
    return size;
}

constexpr bool is_whitespace(char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

template<typename ST, typename Publish>
void words_extract_iterative(ST& current_word, Publish publish)
{
    if(!current_word.empty())
    {
        publish(std::move(current_word));
    }
}

template<typename ST,typename SeparatorPred, typename ReconstructCurrentWord, typename Publish>
void words_extract_iterative(ST& current_word, char c, SeparatorPred separator_pred, ReconstructCurrentWord rcw, Publish publish)
{
    if(separator_pred(c))
    {
        if(!current_word.empty())
        {
            if(publish(std::move(current_word)))
            {
                current_word = rcw();
            }
            else
            {
                current_word.clear();
            }
        }
    }
    else
    {
        current_word.push_back(c);
    }
}

template<typename T>
using allocator_base = std::pmr::polymorphic_allocator<T>;

int main(int argc, char* argv[])
{
    using seconds_double = std::chrono::duration<double, std::ratio<1,1>>;
    using char_allocator_type = allocator_base<char>;
    using string_type = basic_string<char, char_traits<char>, char_allocator_type>;
    using string_view_type = string_view;
    using pair_allocator_type = allocator_base<pair<string_view , uint64_t>>;
    using const_pair_allocator_type = allocator_base<pair<const string_type, uint64_t>>;
    using unordered_map_type = unordered_map<string_type, uint64_t, hash<string_type>, std::equal_to<>, const_pair_allocator_type>;
    using vector_type = vector<pair<string_view_type, uint64_t>, pair_allocator_type>;
    using sequential_buffered_stream_type = sequential_buffered_stream<char, char_traits<char>, char_allocator_type>;

    std::filesystem::path file_name = "gutenberg-500M.txt";

    if(argc >= 2)
    {
        file_name = argv[1];
    }

    ifstream file_stream;
    file_stream.open(file_name);
    if(!file_stream.is_open())
    {
        cerr << "Couldn't find " << file_name << '!' << endl;
        return -1;
    }

    auto fsize = file_size(file_stream);
    auto buffer_resource_size = fsize * 128_MB / 512_MB;
    auto initial_buckets = fsize * 2_KB / 512_MB;
    auto buffer_size = 32_MB;

    pmr::monotonic_buffer_resource buffer_resource{buffer_resource_size};
    char_allocator_type char_allocator{&buffer_resource};
    const_pair_allocator_type const_pair_allocator{&buffer_resource};
    pair_allocator_type pair_allocator{&buffer_resource};

    {
        sequential_buffered_stream_type input_stream(file_stream, buffer_size, char_allocator);

        istream& in = input_stream;

        {
            vector<char, char_allocator_type> buffer_for_it(buffer_size, char_allocator);

            istreambuf_buffered_iterator<char> is_begin{in.rdbuf(), buffer_for_it.data(), buffer_for_it.data() + buffer_for_it.size()}, is_end;

            // read words
            auto start = chrono::high_resolution_clock::now();
            unordered_map_type word_occurences(initial_buckets, const_pair_allocator);
            auto publish = [&word_occurences](string_type&& word)
            {
                auto fit = word_occurences.find(word);
                if(fit != std::end(word_occurences))
                {
                    ++fit->second;
                    return false;
                }
                else
                {
                    word_occurences.emplace(std::make_pair(std::move(word), uint64_t(1)));
                    return true;
                }
            };
            auto recreate_string = [&char_allocator]()
            {
                string_type word{char_allocator};
                word.reserve(64);
                return word;
            };

            string_type current_word = recreate_string();
            std::for_each(is_begin, is_end, [&current_word, &recreate_string, &publish](char c){
                words_extract_iterative(current_word, c, is_whitespace, recreate_string, publish);
            });
            words_extract_iterative(current_word, publish);

            vector_type words(word_occurences.size(), pair_allocator);
            {
                std::transform(std::begin(word_occurences), std::end(word_occurences), std::begin(words), [](auto&& p){
                    return std::make_pair(string_view{p.first}, p.second);
                });

                std::sort(std::begin(words), std::end(words), [](auto&& l, auto&& r){ return l.second>r.second; });
            }
            auto end = chrono::high_resolution_clock::now();
            cout << "" << chrono::duration_cast<seconds_double>(end - start).count() << 's' << endl;

            // print
            //std::for_each(std::begin(words), std::begin(words) + 10, [](auto&& p){
            //    cout << p.first << ": " << p.second << endl;
            //});

            //size_t sum = 0;
            //for(auto& wp : words)
            //{
            //    sum += wp.second;
            //}
            //cout << sum << endl;
        }
    }

    return 0;
}
