#include "x11/cuda_x11.h"

void quark_blake512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_nonceVector, uint32_t *d_outputHash, int order);

// ---- optimised but non compatible kernels

void x16_echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);

// ---- 80 bytes kernels

void qubit_luffa512_cpu_init(int thr_id, uint32_t threads);


void x16_echo512_cuda_init(int thr_id, const uint32_t threads);