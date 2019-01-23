#ifndef START_FILE_HANDLER_H
#define START_FILE_HANDLER_H

#include <vector>
#include "solverstructs.h"

class start_file_handler
{
public:
    static std::vector<start_condition> load_all();
    static bool save_all(const std::vector<start_condition_t> data);
};

#endif // START_FILE_HANDLER_H
