#include <iostream>
#include <string>
#include <string_view>
using namespace std;

template<typename F>
void split(string_view str, const string& delims, F f)
{
    size_t prev = 0, pos;
    while ((pos = str.find_first_of(" ';", prev)) != string::npos)
    {
        if (pos > prev)
            f(str.substr(prev, pos-prev));
        prev = pos+1;
    }
    if (prev < str.length())
        f(str.substr(prev, string::npos));
}

int main(int argc, char* argv[])
{
    string line;
    while (getline(std::cin, line))
    {
        split(line, "\r\t ", [](string_view word){ cout << word << ' ' << 1 << '\n';});
    }
    cout.flush();
}