#ifndef RESULT_FILE_H
#define RESULT_FILE_H

#include <cstdio>
#include <cstdint>
#include <string>

class result_file
{
public:
    static uint64_t load(const std::string& filename);
    static bool save(uint64_t res, const std::string& filename);
    static bool save(uint64_t res, FILE* file);
};

#endif // RESULT_FILE_H
