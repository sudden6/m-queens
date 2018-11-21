#include "filereader.h"
#include <iostream>

FileReader::FileReader(const std::string& filename)
{
    file.open(filename, std::ifstream::in | std::ifstream::binary);
}

bool FileReader::is_open()
{
    return file.is_open();
}

std::vector<FileReader::Record> FileReader::getNext(size_t count)
{
    std::vector<Record> result;
    result.resize(count);

    uint8_t data[22] = {0};
    char* data_p = reinterpret_cast<char*> (data);
    size_t i = 0;
    for(; i < count; i++) {
        file.read(data_p, 22);

        if(file.eof()) {
            break;
        }

        if(!file) {
            std::cout << "Error reading file" << std::endl;
            break;
        }

        Record& rec = result[i];

        rec.hor  = static_cast<uint32_t>(data[0]) << 8*3
                 | static_cast<uint32_t>(data[1]) << 8*2
                 | static_cast<uint32_t>(data[2]) << 8*1
                 | static_cast<uint32_t>(data[3]) << 8*0;

        rec.vert = static_cast<uint32_t>(data[4]) << 8*3
                 | static_cast<uint32_t>(data[5]) << 8*2
                 | static_cast<uint32_t>(data[6]) << 8*1
                 | static_cast<uint32_t>(data[7]) << 8*0;

        rec.diag_up   = static_cast<uint64_t>(data[8])  << 8*6
                      | static_cast<uint64_t>(data[9])  << 8*5
                      | static_cast<uint64_t>(data[10]) << 8*4
                      | static_cast<uint64_t>(data[11]) << 8*3
                      | static_cast<uint64_t>(data[12]) << 8*2
                      | static_cast<uint64_t>(data[13]) << 8*1
                      | static_cast<uint64_t>(data[14]) << 8*0;

        rec.diag_down = static_cast<uint64_t>(data[15]) << 8*6
                      | static_cast<uint64_t>(data[16]) << 8*5
                      | static_cast<uint64_t>(data[17]) << 8*4
                      | static_cast<uint64_t>(data[18]) << 8*3
                      | static_cast<uint64_t>(data[19]) << 8*2
                      | static_cast<uint64_t>(data[20]) << 8*1
                      | static_cast<uint64_t>(data[21]) << 8*0;

        if(file.eof()) {
            count++;
            break;
        }
    }

    result.resize(i);
    return result;
}

