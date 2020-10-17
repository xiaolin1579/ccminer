/**
 * This code compares final hash against target
 */
 #include <stdio.h>
 #include <memory.h>
 
 #include "miner.h"
 
 #include "cuda_helper.h"
 #include "cuda_x11.h"
 

__constant__ uint32_t pTarget[8]; // 32 bytes
 
 // store MAX_GPUS device arrays of 8 nonces
 static uint32_t* h_resNonces[MAX_GPUS] = { NULL };
 static uint32_t* d_resNonces[MAX_GPUS] = { NULL };
 static __thread bool init_done = false;
 
 /* --------------------------------------------------------------------------------------------- */
 
__host__
int cuda_process_hash_64(const int thr_id, const uint32_t throughput, uint32_t nonce, uint32_t *d_outputHash, int order, const int index)
{
	switch (index)
	{
		case 0:
			quark_blake512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_outputHash, order++);
			break;
		case 1:
			quark_bmw512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_outputHash, order++);
			break;
		case 2:
			quark_groestl512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_outputHash, order++);
			break;
		case 3:
			quark_skein512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_outputHash, order++);
			break;
		case 4:
			quark_jh512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_outputHash, order++);
			break;
		case 5:
			quark_keccak512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_outputHash, order++);
			break;
		case 6:
			x11_luffa512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_outputHash, order++);
			break;
		case 7:
			x11_cubehash512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_outputHash, order++);
			break;
		case 8:
			x11_shavite512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_outputHash, order++);
			break;
		case 9:
			x11_simd512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_outputHash, order++);
			break;
		case 10:
			x11_echo512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_outputHash, order++);
			break;
	}

	return order;
}

__host__
int cuda_x11k_hash(const int thr_id, const uint32_t throughput, uint32_t nonce, uint32_t *d_outputHash, int order, const int number_of_iterations, const int number_of_algos)
{
	static unsigned char *index_seed;
	uint32_t _ALIGN(64) vhash[8];

	CUDA_CALL_OR_RET_X(cudaMalloc(&index_seed, sizeof(unsigned char)), 0);

	// Iteration 0
	quark_blake512_cpu_hash_80(thr_id, throughput, nonce, d_outputHash); order++;

	cudaMemcpy(index_seed, (unsigned char *) d_outputHash, sizeof(unsigned char), cudaMemcpyDeviceToDevice);

	for (int i = 1; i < number_of_iterations; i++)
	{	
		int index = (index_seed[i]) % number_of_algos;
		order = cuda_process_hash_64(thr_id, throughput, nonce, d_outputHash, order, index);
	}

	return order;
}
