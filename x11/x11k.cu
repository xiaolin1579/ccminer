extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"
#include "sph/sph_luffa.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"
}

#include "miner.h"
#include "cuda_helper.h"
#include "cuda_x11.h"

#include <stdio.h>
#include <memory.h>

static uint32_t *d_hash[MAX_GPUS];

const uint64_t GetUint64(const uint8_t *data, int pos)
{
	const uint8_t *ptr = data + pos * 8;
	return ((uint64_t)ptr[0]) | \
			((uint64_t)ptr[1]) << 8 | \
			((uint64_t)ptr[2]) << 16 | \
			((uint64_t)ptr[3]) << 24 | \
			((uint64_t)ptr[4]) << 32 | \
			((uint64_t)ptr[5]) << 40 | \
			((uint64_t)ptr[6]) << 48 | \
			((uint64_t)ptr[7]) << 56;
}

void *Blake512(void *oHash, const void *iHash)
{
	sph_blake512_context ctx_blake;

	sph_blake512_init(&ctx_blake);
	sph_blake512 (&ctx_blake, iHash, 80);
	sph_blake512_close (&ctx_blake, oHash);
}

void *Bmw512(void *oHash, const void *iHash)
{
	sph_bmw512_context ctx_bmw;

	sph_bmw512_init(&ctx_bmw);
	sph_bmw512 (&ctx_bmw, iHash, 64);
	sph_bmw512_close(&ctx_bmw, oHash);
}

void *Groestl512(void *oHash, const void *iHash)
{
	sph_groestl512_context ctx_groestl;

	sph_groestl512_init(&ctx_groestl);
	sph_groestl512 (&ctx_groestl, iHash, 64);
	sph_groestl512_close(&ctx_groestl, oHash);
}

void *Skein512(void *oHash, const void *iHash)
{
	sph_skein512_context ctx_skein;

	sph_skein512_init(&ctx_skein);
	sph_skein512 (&ctx_skein, iHash, 64);
	sph_skein512_close (&ctx_skein, oHash);
}

void *Jh512(void *oHash, const void *iHash)
{
	sph_jh512_context ctx_jh;

	sph_jh512_init(&ctx_jh);
	sph_jh512 (&ctx_jh, iHash, 64);
	sph_jh512_close(&ctx_jh, oHash);
}

void *Keccak512(void *oHash, const void *iHash)
{
	sph_keccak512_context ctx_keccak;

	sph_keccak512_init(&ctx_keccak);
	sph_keccak512 (&ctx_keccak, iHash, 64);
	sph_keccak512_close(&ctx_keccak, oHash);
}

void *Luffa512(void *oHash, const void *iHash)
{
	sph_luffa512_context ctx_luffa1;

	sph_luffa512_init (&ctx_luffa1);
	sph_luffa512 (&ctx_luffa1, iHash, 64);
	sph_luffa512_close (&ctx_luffa1, oHash);
}

void *Cubehash512(void *oHash, const void *iHash)
{
	sph_cubehash512_context	ctx_cubehash1;

	sph_cubehash512_init (&ctx_cubehash1);
	sph_cubehash512 (&ctx_cubehash1, iHash, 64);
	sph_cubehash512_close(&ctx_cubehash1, oHash);
}

void *Shavite512(void *oHash, const void *iHash)
{
	sph_shavite512_context ctx_shavite1;

	sph_shavite512_init (&ctx_shavite1);
	sph_shavite512 (&ctx_shavite1, iHash, 64);
	sph_shavite512_close(&ctx_shavite1, oHash);
}

void *Simd512(void *oHash, const void *iHash)
{
	sph_simd512_context	ctx_simd1;

	sph_simd512_init (&ctx_simd1);
	sph_simd512 (&ctx_simd1, iHash, 64);
	sph_simd512_close(&ctx_simd1, oHash);
}

void *Echo512(void *oHash, const void *iHash)
{
	sph_echo512_context	ctx_echo1;

	sph_echo512_init (&ctx_echo1);
	sph_echo512 (&ctx_echo1, iHash, 64);
	sph_echo512_close(&ctx_echo1, oHash);
}

/*
void *fnHashX11K[] = {
	Blake512,
	Bmw512,
	Groestl512,
	Skein512,
	Jh512,
	Keccak512,
	Luffa512,
	Cubehash512,
	Shavite512,
	Simd512,
	Echo512,
};
*/

void processHash(void *oHash, const void *iHash, const int index)
{
	/*
	void (*hashX11k)(void *oHash, const void *iHash);

	hashX11k = fnHashX11K[index];
	(*hashX11k)(oHash, iHash);
	*/
	switch(index)
	{
		case 0:
			Blake512(oHash, iHash);
			break;

		case 1:
			Bmw512(oHash, iHash);
			break;

		case 2:
			Groestl512(oHash, iHash);
			break;

		case 3:
			Skein512(oHash, iHash);
			break;

		case 4:
			Jh512(oHash, iHash);
			break;

		case 5:
			Keccak512(oHash, iHash);
			break;

		case 6:
			Luffa512(oHash, iHash);
			break;

		case 7:
			Cubehash512(oHash, iHash);
			break;

		case 8:
			Shavite512(oHash, iHash);
			break;

		case 9:
			Simd512(oHash, iHash);
			break;

		case 10:
			Echo512(oHash, iHash);

		default:
			break;
	}
}



// X11K CPU Hash
extern "C" void x11khash(void *output, const void *input)
{
	const int HASHX11K_NUMBER_ITERATIONS = 64;

	unsigned char _ALIGN(128) hashA[128] = { 0 };
	unsigned char _ALIGN(128) hashB[128] = { 0 };
	//void* hashA = (void*) malloc(64);
	//void* hashB = (void*) malloc(64);

	// Iteration 0
	processHash(hashA, input, 0);

	for(int i = 1; i < HASHX11K_NUMBER_ITERATIONS; i++) {
		uint64_t index = GetUint64(hashA, i % 8) % 11;
		processHash(hashB, hashA, index);
		memcpy(hashA, hashB, 64);

	//    void* t = hashA;
		//hashA = hashB;
		//hashB = t;
	}

	memcpy(output, hashA, 32);

	//free(hashA);
	//free(hashB);

}

//#define _DEBUG
#define _DEBUG_PREFIX "x11k"
#include "cuda_debug.cuh"

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_x11k(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	int intensity = (device_sm[device_map[thr_id]] >= 500 && !is_windows()) ? 20 : 19;
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity); // 19=256*256*8;
	//if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		ptarget[7] = 0x5;

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		quark_blake512_cpu_init(thr_id, throughput);
		quark_bmw512_cpu_init(thr_id, throughput);
		quark_groestl512_cpu_init(thr_id, throughput);
		quark_skein512_cpu_init(thr_id, throughput);
		quark_keccak512_cpu_init(thr_id, throughput);
		quark_jh512_cpu_init(thr_id, throughput);
		x11_luffaCubehash512_cpu_init(thr_id, throughput);
		x11_shavite512_cpu_init(thr_id, throughput);
		x11_echo512_cpu_init(thr_id, throughput);
		if (x11_simd512_cpu_init(thr_id, throughput) != 0) {
			return 0;
		}
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput), 0);

		cuda_check_cpu_init(thr_id, throughput);

		init[thr_id] = true;
	}

	uint32_t endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	quark_blake512_cpu_setBlock_80(thr_id, endiandata);
	cuda_check_cpu_setTarget(ptarget);

	do {
		int order = 0;

		// Hash with CUDA
		quark_blake512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
		TRACE("blake  :");
		quark_bmw512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		TRACE("bmw    :");
		quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		TRACE("groestl:");
		quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		TRACE("skein  :");
		quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		TRACE("jh512  :");
		quark_keccak512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		TRACE("keccak :");
		x11_luffaCubehash512_cpu_hash_64(thr_id, throughput, d_hash[thr_id], order++);
		TRACE("luffa+c:");
		x11_shavite512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		TRACE("shavite:");
		x11_simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		TRACE("simd   :");
		x11_echo512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		TRACE("echo => ");

		*hashes_done = pdata[19] - first_nonce + throughput;

		work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
		if (work->nonces[0] != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], work->nonces[0]);
			x11khash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				work->nonces[1] = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
				if (work->nonces[1] != 0) {
					be32enc(&endiandata[19], work->nonces[1]);
					x11khash(vhash, endiandata);
					bn_set_target_ratio(work, vhash, 1);
					work->valid_nonces++;
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				} else {
					pdata[19] = work->nonces[0] + 1; // cursor
				}
				return work->valid_nonces;
			} else {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
				pdata[19] = work->nonces[0] + 1;
				continue;
			}
		}

		if ((uint64_t) throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}
		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

// cleanup
extern "C" void free_x11k(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	cudaFree(d_hash[thr_id]);

	quark_blake512_cpu_free(thr_id);
	quark_groestl512_cpu_free(thr_id);
	x11_simd512_cpu_free(thr_id);

	cuda_check_cpu_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}
