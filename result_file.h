#ifndef RESULT_FILE_H
#define RESULT_FILE_H

#include <cstdint>

class result_file
{
public:
    static uint64_t load();
    static bool save(uint64_t res);
};

#endif // RESULT_FILE_H
