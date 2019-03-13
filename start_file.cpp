#include "start_file.h"
#include "serialize_util.h"
#include <string>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <regex>

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

// expects an opened file in binary read mode
std::vector<start_condition> start_file::load_all(FILE* file)
{
    if(!file) {
        std::cout << "Error reading file" << std::endl;
        return{};
    }

    constexpr size_t record_size = sizeof(start_condition_t);
    uint8_t data[record_size] = {0};
    std::vector<start_condition_t> res;

    while (true) {
        if(feof(file)) {
            break;
        }
        size_t cnt = fread(data, record_size, 1, file);
        if(cnt != 1) {
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

start_file::file_info start_file::parse_filename(const std::string& filename) {
    file_info fi;
    const std::regex parser{R"(N_(\d+)_D_(\d+)_(\d+)_(\d+)\.pre$)"};
    std::smatch matches;

    if (!std::regex_match(filename, matches, parser) || matches.size() != 5) {
        std::cout << "Failed to parse filename: " << filename << std::endl;
        return fi;
    }

    // parse placed
    {
        std::string placed_str = matches[2].str();
        unsigned long placed_l = 0;
        bool fail = true;
        try {
            placed_l = std::stoul(placed_str);
            fail = placed_l > std::numeric_limits<uint8_t>::max();
        } catch (...) {
        }

        if(fail) {
            std::cout << "Failed to parse placed" << std::endl;
            return fi;
        }
        fi.placed = static_cast<uint8_t>(placed_l);
    }

    // parse start_idx
    {
        std::string start_idx_str = matches[3].str();
        unsigned long long start_idx_l = 0;
        bool fail = true;
        try {
            start_idx_l = std::stoull(start_idx_str);
            fail = start_idx_l > std::numeric_limits<uint64_t>::max();
        } catch (...) {
        }

        if(fail) {
            std::cout << "Failed to parse start_idx" << std::endl;
            return fi;
        }
        fi.start_idx = static_cast<uint64_t>(start_idx_l);
    }

    // parse end_idx
    {
        std::string end_idx_str = matches[4].str();
        unsigned long long end_idx_l = 0;
        bool fail = true;
        try {
            end_idx_l = std::stoull(end_idx_str);
            fail = end_idx_l > std::numeric_limits<uint64_t>::max();
        } catch (...) {
        }

        if(fail) {
            std::cout << "Failed to parse end_idx" << std::endl;
            return fi;
        }
        fi.end_idx = static_cast<uint64_t>(end_idx_l);
    }

    // parse boardsize
    // must be last, because if boardsize is valid file_info is valid
    {
        std::string boardsize_str = matches[1].str();
        unsigned long boardsize_l = 0;
        bool fail = true;
        try {
            boardsize_l = std::stoul(boardsize_str);
            fail = boardsize_l > std::numeric_limits<uint8_t>::max() || boardsize_l < 1;
        } catch (...) {
        }

        if(fail) {
            std::cout << "Failed to parse boardsize" << std::endl;
            return fi;
        }
        fi.boardsize = static_cast<uint8_t>(boardsize_l);
    }

    return fi;
}

