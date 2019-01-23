#include "serialize_util.h"

void serialize_util::pack_u32(uint32_t i, uint8_t* o)
{
    o[0] = i;
    o[1] = i >> 8;
    o[2] = i >> 16;
    o[3] = i >> 24;
}

uint32_t serialize_util::unpack_u32(const uint8_t* o)
{
    return o[0] | o[1] << 8 | o[2] << 16 | o[3] << 24;
}
