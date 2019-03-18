#ifndef RESULT_FILE_H
#define RESULT_FILE_H

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

class result_file
{
public:
    static bool save(std::vector<uint64_t> res, const std::string& filename);
    static bool save(std::vector<uint64_t> res, FILE* file);
};

#endif // RESULT_FILE_H
