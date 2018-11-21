#ifndef FILEREADER_H
#define FILEREADER_H

#include <fstream>
#include <string>
#include <vector>

class FileReader
{
    std::ifstream file;
public:
    struct Record {
        uint64_t diag_up;
        uint64_t diag_down;
        uint32_t hor;
        uint32_t vert;
    };

    FileReader(const std::string &filename);
    bool is_open();

    std::vector<Record> getNext(size_t count = 1);


};

#endif // FILEREADER_H
