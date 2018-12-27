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

    std::vector<Record> getNext(size_t count = 1);


};

#endif // FILEREADER_H
