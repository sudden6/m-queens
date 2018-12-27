#include "filereader.h"
#include <cstring>
#include <iostream>

FileReader::FileReader(const std::string& filename)
{
    file.open(filename, std::ifstream::in | std::ifstream::binary);
}

bool FileReader::is_open()
{
    return file.is_open();
}

std::vector<Preplacement> FileReader::getNext(size_t count)
{
    std::vector<Preplacement> result;
    result.resize(count);

    uint8_t data[record_size] = {0};
    char* data_p = reinterpret_cast<char*> (data);
    size_t i = 0;
    for(; i < count; i++) {
        file.read(data_p, record_size);

        if(file.eof()) {
            break;
        }

        if(!file) {
            std::cout << "Error reading file" << std::endl;
            break;
        }

        Preplacement& rec = result[i];

        std::memcpy(rec.raw, data, record_size);

        if(file.eof()) {
            count++;
            break;
        }
    }

    result.resize(i);
    return result;
}

