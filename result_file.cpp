#include "result_file.h"
#include "serialize_util.h"

#include <string>
#include <iostream>
#include <fstream>

bool result_file::save(std::vector<uint64_t> res, const std::string& filename)
{
    std::ofstream file;
    file.open(filename, std::ofstream::out | std::ofstream::binary);

    if(!file || !file.is_open()) {
        std::cout << "Error writing file" << std::endl;
        return false;
    }

    constexpr size_t record_size = sizeof(uint64_t);
    uint8_t record[record_size] = {0};
    char* data_p = reinterpret_cast<char*> (record);
    for(const auto& element : res) {
        serialize_util::pack_u64(element, record);

        file.write(data_p, record_size);

        if(!file) {
            return false;
        }
    }

    return true;
}

// expects a write binary opened file
bool result_file::save(std::vector<uint64_t> res, FILE *file)
{
    if(!file) {
        std::cout << "Error writing file" << std::endl;
        return false;
    }

    constexpr size_t record_size = sizeof(uint64_t);
    uint8_t record[record_size] = {0};

    for(const auto& element : res) {
        serialize_util::pack_u64(element, record);

        size_t cnt = fwrite(record, record_size, 1, file);

        if(cnt != 1) {
            return false;
        }
    }

    return true;
}
