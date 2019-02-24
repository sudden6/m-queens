#include "start_file.h"
#include "serialize_util.h"
#include "boinc/boinc_api.h"
#include <string>
#include <cstdio>
#include <iostream>
#include <fstream>

std::vector<start_condition> start_file::load_all(const std::string& filename)
{
    std::ifstream file;
    file.open(filename, std::ifstream::in | std::ifstream::binary);

    if(!file || !file.is_open()) {
        std::cout << "Error reading file" << std::endl;
        return{};
    }

    constexpr size_t record_size = sizeof(start_condition_t);
    uint8_t data[record_size] = {0};
    char* data_p = reinterpret_cast<char*> (data);
    std::vector<start_condition_t> res;

    while (file) {
        if(file.peek() == EOF) {
            break;
        }
        file.read(data_p, record_size);
        if(!file) {
            std::cout << "Incomplete record in file" << std::endl;
            return {};
        }

        start_condition_t start;
        start.cols  = serialize_util::unpack_u32(&data[0]);
        start.diagl = serialize_util::unpack_u32(&data[4]);
        start.diagr = serialize_util::unpack_u32(&data[8]);

        res.push_back(start);
    }

    return res;
}

bool start_file::save_all(const std::vector<start_condition_t> data, const std::string &filename)
{
    std::ofstream file;
    file.open(filename, std::ofstream::out | std::ofstream::binary);

    if(!file || !file.is_open()) {
        std::cout << "Error writing file" << std::endl;
        return false;
    }

    constexpr size_t record_size = sizeof(start_condition_t);
    uint8_t record[record_size] = {0};
    char* data_p = reinterpret_cast<char*> (record);
    for(const auto& element : data) {
        serialize_util::pack_u32(element.cols,  &record[0]);
        serialize_util::pack_u32(element.diagl, &record[4]);
        serialize_util::pack_u32(element.diagr, &record[8]);

        file.write(data_p, record_size);

        if(!file) {
            return false;
        }
    }

    return true;
}
