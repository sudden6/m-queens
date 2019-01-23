#include "serialize_util.h"

void serialize_util::pack_u16(uint16_t i, uint8_t *o)
{
    o[0] = i;
    o[1] = i >> 8;
}

uint16_t serialize_util::unpack_u16(const uint8_t* o)
{
    return static_cast<uint16_t>(o[0]) | static_cast<uint16_t>(o[1]) << 8;
}

void serialize_util::pack_u32(uint32_t i, uint8_t* o)
{
    pack_u16(i, &o[0]);
    pack_u16(i >> 16, &o[2]);
}

uint32_t serialize_util::unpack_u32(const uint8_t* o)
{
    return static_cast<uint32_t>(unpack_u16(&o[0])) |
           static_cast<uint32_t>(unpack_u16(&o[2])) << 16;
}

void serialize_util::pack_u64(uint64_t i, uint8_t *o)
{
    pack_u32(i, &o[0]);
    pack_u32(i >> 32, &o[4]);
}

uint64_t serialize_util::unpack_u64(const uint8_t *o)
{
    return static_cast<uint64_t>(unpack_u32(&o[0])) |
           static_cast<uint64_t>(unpack_u32(&o[4])) << 32;
}
