#include <parallel_primitives.hpp>
#include <string>

namespace parallel
{

    using std::literals::operator""s;

    std::ostream& operator<<(std::ostream& os, const static_schedule& sch)
    {
        os << (sch.chunk_size ? "static["s + std::to_string(*sch.chunk_size) + ']' : "static");
        return os;
    }

    std::ostream& operator<<(std::ostream& os, const dynamic_schedule& sch)
    {
        os << (sch.chunk_size ? "dynamic["s + std::to_string(*sch.chunk_size) + ']' : "dynamic");
        return os;
    }

    std::ostream& operator<<(std::ostream& os, const guided_schedule& sch)
    {
        os << (sch.chunk_size ? "guided["s + std::to_string(*sch.chunk_size) + ']' : "guided");
        return os;
    }

    std::ostream& operator<<(std::ostream& os, const auto_schedule& sch)
    {
        os << "auto";
        return os;
    }

    std::ostream& operator<<(std::ostream& os, const runtime_schedule& sch)
    {
        os << "runtime";
        return os;
    }

    std::ostream& operator<<(std::ostream& os, const schedule_t& v)
    {
        std::visit([&os](auto&& arg) {
            os << arg;
            }, v);
        return os;
    }
}
