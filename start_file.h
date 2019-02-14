#ifndef START_FILE_HANDLER_H
#define START_FILE_HANDLER_H

#include <string>
#include <vector>
#include "solverstructs.h"

class start_file
{
public:
    static std::vector<start_condition> load_all(const std::string &filename);
    static bool save_all(const std::vector<start_condition_t> data, const std::string &filename);
};

#endif // START_FILE_HANDLER_H
