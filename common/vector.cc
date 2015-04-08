#include "vector.h"
#include <algorithm>

float4_t* float4_alloc(std::size_t n) {
    void* tmp = 0;
    std::size_t align = std::max(sizeof(float4_t), sizeof(void*));
    std::size_t size = sizeof(float4_t);
    size *= static_cast<std::size_t>(n);
    if (posix_memalign(&tmp, align, size)) {
        throw std::bad_alloc();
    }
    return static_cast<float4_t*>(tmp);
}

float8_t* float8_alloc(std::size_t n) {
    void* tmp = 0;
    std::size_t align = std::max(sizeof(float8_t), sizeof(void*));
    std::size_t size = sizeof(float8_t);
    size *= static_cast<std::size_t>(n);
    if (posix_memalign(&tmp, align, size)) {
        throw std::bad_alloc();
    }
    return static_cast<float8_t*>(tmp);
}
