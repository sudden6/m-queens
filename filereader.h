#ifndef FILEREADER_H
#define FILEREADER_H

#include <fstream>
#include <string>
#include <vector>

#include "solverstructs.h"

class FileReader
{
    std::ifstream file;
public:
    FileReader(const std::string &filename);
    bool is_open();

    std::vector<Preplacement> getNext(size_t count = 1);
    static constexpr size_t record_size = 8;
    static constexpr uint8_t empty = 0xFF;
};

#endif // FILEREADER_H
