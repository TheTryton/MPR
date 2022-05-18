#include <iostream>
#include <string>
#include <string_view>
using namespace std;

int main(int argc, char* argv[])
{
    string current_word{};
    size_t current_count = 0;
    string line;
    string word;
    while (getline(std::cin, line))
    {
        string_view str = line;
        auto delim = str.find(' ');
        word = str.substr(0, delim);
        if(word == current_word)
        {
            ++current_count;
        }
        else
        {
            if(!current_word.empty())
            {
                cout << current_word << ' ' << current_count << '\n';
            }
            current_word = word;
            current_count = 1;
        }
    }
    if(current_word == word)
    {
        cout << current_word << ' ' << current_count << '\n';
    }
    cout.flush();
}