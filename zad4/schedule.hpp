#pragma once

#include <optional>
#include <variant>
#include <ostream>

struct static_schedule{
    std::optional<size_t> chunk_size;
    friend std::ostream& operator<<(std::ostream& os, const static_schedule& sch);
};
struct dynamic_schedule{
    std::optional<size_t> chunk_size;
    friend std::ostream& operator<<(std::ostream& os, const dynamic_schedule& sch);
};
struct guided_schedule{
    std::optional<size_t> chunk_size;
    friend std::ostream& operator<<(std::ostream& os, const guided_schedule& sch);
};
struct auto_schedule{
    friend std::ostream& operator<<(std::ostream& os, const auto_schedule& sch);
};
struct runtime_schedule{
    friend std::ostream& operator<<(std::ostream& os, const runtime_schedule& sch);
};

using schedule_t = std::variant<static_schedule, dynamic_schedule, guided_schedule, auto_schedule, runtime_schedule>;

std::ostream& operator<<(std::ostream& os, const schedule_t& v);