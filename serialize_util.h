#ifndef SERIALIZE_UTIL_H
#define SERIALIZE_UTIL_H

#include <cstdint>

class serialize_util
{
public:
    static void pack_u32(uint32_t i, uint8_t* o);
    static uint32_t unpack_u32(const uint8_t* o);


    static void pack_u64(uint64_t i, uint8_t* o);
    static uint64_t unpack_u64(const uint8_t* o);
};

#endif // SERIALIZE_UTIL_H
