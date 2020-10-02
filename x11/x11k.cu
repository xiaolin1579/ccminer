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

extern "C" static void processHash(void *oHash, const void *iHash, const int index, const size_t len)
{
      switch (index)
      {
      case 0:
        sph_blake512_context ctx_blake;

	sph_blake512_init(&ctx_blake);
	sph_blake512(&ctx_blake, iHash, len);
	sph_blake512_close(&ctx_blake, oHash);
        break;
      case 1:
        sph_bmw512_context ctx_bmw;

        sph_bmw512_init(&ctx_bmw);
        sph_bmw512 (&ctx_bmw, iHash, len);
        sph_bmw512_close(&ctx_bmw, oHash);
         break;
      case 2:
        sph_groestl512_context ctx_groestl;

        sph_groestl512_init(&ctx_groestl);
        sph_groestl512 (&ctx_groestl, iHash, len);
        sph_groestl512_close(&ctx_groestl, oHash);
         break;
      case 3:
        sph_skein512_context ctx_skein;

        sph_skein512_init(&ctx_skein);
        sph_skein512 (&ctx_skein, iHash, len);
        sph_skein512_close (&ctx_skein, oHash);
         break;
      case 4:
        sph_jh512_context ctx_jh;

        sph_jh512_init(&ctx_jh);
        sph_jh512 (&ctx_jh, iHash, len);
        sph_jh512_close(&ctx_jh, oHash);
         break;
      case 5:
        sph_keccak512_context ctx_keccak;

        sph_keccak512_init(&ctx_keccak);
        sph_keccak512 (&ctx_keccak, iHash, len);
        sph_keccak512_close(&ctx_keccak, oHash);
         break;
      case 6:
        sph_luffa512_context ctx_luffa1;

        sph_luffa512_init (&ctx_luffa1);
        sph_luffa512 (&ctx_luffa1, iHash, len);
        sph_luffa512_close (&ctx_luffa1, oHash);
         break;
      case 7:
        sph_cubehash512_context ctx_cubehash1;

        sph_cubehash512_init (&ctx_cubehash1);
        sph_cubehash512 (&ctx_cubehash1, iHash, len);
        sph_cubehash512_close(&ctx_cubehash1, oHash);
         break;
      case 8:
        sph_shavite512_context ctx_shavite1;

        sph_shavite512_init (&ctx_shavite1);
        sph_shavite512 (&ctx_shavite1, iHash, len);
        sph_shavite512_close(&ctx_shavite1, oHash);
         break;
      case 9:
        sph_simd512_context ctx_simd1;

        sph_simd512_init (&ctx_simd1);
        sph_simd512 (&ctx_simd1, iHash, len);
        sph_simd512_close(&ctx_simd1, oHash);
         break;
      case 10:
        sph_echo512_context ctx_echo1;

        sph_echo512_init (&ctx_echo1);
        sph_echo512 (&ctx_echo1, iHash, len);
        sph_echo512_close(&ctx_echo1, oHash);
         break;
      }
}



// X11K CPU Hash
extern "C" void x11khash(void *output, const void *input)
{
        const int HASHX11K_NUMBER_ITERATIONS = 64;
        const int HASHX11K_NUMBER_ALGOS = 11;

	//unsigned char _ALIGN(64) hash[128];
	uint32_t _ALIGN(64) hashA[16], hashB[16];
	//uint32_t _ALIGN(64) hashA[128], hashB[128];
	
        // Iteration 0
        processHash(hashA, input, 0, 80);

        for(int i = 1; i < HASHX11K_NUMBER_ITERATIONS; i++) {
		uint32_t * p = (uint32_t *) hashA;
                processHash(hashB, hashA, p[i] % HASHX11K_NUMBER_ALGOS, 64);

		memcpy(hashA, hashB, 64);
        }

        memcpy(output, hashA, 32);
}

//#define _DEBUG
#define _DEBUG_PREFIX "x11k"
#include "cuda_debug.cuh"

static bool init[MAX_GPUS] = { 0 };

extern void quark_blake512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_outputHash, int order);

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
		quark_blake512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]);
		TRACE("blake  :");

		gpulog(LOG_INFO, thr_id, "before for loop");
		   for (int i = 1; i < 64; i++)
		   {
			gpulog(LOG_INFO, thr_id, "before setting p");
			gpulog(LOG_INFO, thr_id, "d_hash[thr_id] is %c", d_hash[thr_id]);

			uint32_t * p = (uint32_t *) d_hash[thr_id];

			gpulog(LOG_INFO, thr_id, "before switch");
			gpulog(LOG_INFO, thr_id, "p is %c", p);

			switch (p[i] % 11)
			{

                        case 0:
				gpulog(LOG_INFO, thr_id, "before blake");
                                quark_blake512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                                TRACE("blake  :");
                                break;
                        case 1:
				gpulog(LOG_INFO, thr_id, "before quark");
                                quark_bmw512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                                TRACE("bmw    :");
                                break;
                        case 2:
				gpulog(LOG_INFO, thr_id, "before groestl");
                                quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                                TRACE("groestl:");
                                break;
                        case 3:
				gpulog(LOG_INFO, thr_id, "before skein");
                                quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                                TRACE("skein  :");
                                break;
                        case 4:
				gpulog(LOG_INFO, thr_id, "before jh");
                                quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                                TRACE("jh512  :");
                                break;
                        case 5:
				gpulog(LOG_INFO, thr_id, "before keccak");
                                quark_keccak512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                                TRACE("keccak :");
                                break;
                        case 6:
				gpulog(LOG_INFO, thr_id, "before luffa");
                                x11_luffa512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                                TRACE("luffa  :");
                                break;
                        case 7:
				gpulog(LOG_INFO, thr_id, "before cubehash");
                                x11_cubehash512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                                TRACE("cube   :");
                                break;
                        case 8:
				gpulog(LOG_INFO, thr_id, "before shavite");
                                x11_shavite512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                                TRACE("shavite:");
                                break;
                        case 9:
				gpulog(LOG_INFO, thr_id, "before simd");
                                x11_simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                                TRACE("simd   :");
                                break;
                        case 10:
				gpulog(LOG_INFO, thr_id, "before echo");
                                x11_echo512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                                TRACE("echo   :");
                                break;
		    }
		}

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
