#include <torch/extension.h>
#include <openssl/sha.h>
#include <iostream>
#include <string.h>
#include <string>
#define MAX_UINT32 4294967295

float hash_single_token(std::string seed, int token) {
    // Setup seed
    unsigned char ibuf[24];
    SHA1((unsigned char *)seed.c_str(), seed.length(), ibuf);

    // Hash
    *(uint32_t *)(ibuf+20) = (uint32_t) token;
    unsigned char obuf[20];
    SHA1(ibuf, 24, obuf);
    uint32_t value = *(uint32_t *)(obuf+16);
    float norm_value = float(value) / MAX_UINT32;

    return norm_value;
}

torch::Tensor hash_tokens_cpu(std::string seed, int vocab_size) {
    // Input seed must be exactly 20 characters long
    torch::Tensor hash_values = torch::empty({vocab_size});

    // Setup seed
    unsigned char ibuf[24];
    SHA1((unsigned char *)seed.c_str(), seed.length(), ibuf);
    for (int i = 0; i < vocab_size; i++)
    {
        *(uint32_t *)(ibuf+20) = (uint32_t) i;
        unsigned char obuf[20];
        SHA1(ibuf, 24, obuf);
        //for (int k = 0; k < 20; k++) { printf("%02x ", obuf[k]); } printf("\n");
        uint32_t value = *(uint32_t *)(obuf+16);
        float norm_value = float(value) / MAX_UINT32;
        hash_values.index_put_({i}, norm_value);
    }

    return hash_values;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("all_index_hash", &hash_tokens_cpu, "Index Range Hash");
    m.def("index_hash", &hash_single_token, "Single Index Hash");
}

