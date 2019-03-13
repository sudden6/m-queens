#ifndef START_FILE_HANDLER_H
#define START_FILE_HANDLER_H

#include <string>
#include <vector>
#include "solverstructs.h"

class start_file
{
public:
    struct file_info {
        uint64_t start_idx = 0;
        uint64_t end_idx = 0;
        uint8_t boardsize = 0;  // if 0, the data is invalid
        uint8_t placed = 0;
    };

    static std::vector<start_condition> load_all(const std::string &filename);
    static bool save_all(const std::vector<start_condition_t> data, const std::string &filename);
    static file_info parse_filename(const std::string &filename);
};

#endif // START_FILE_HANDLER_H
