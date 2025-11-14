//***********************************************************************************
// CUDA Bitcoin Miner - Stage A: Midstate Optimization
// Modified from sequential version for GPU acceleration
//***********************************************************************************

#include <cstdio>
#include <cstring>
#include <cassert>
#include <vector>
#include <algorithm>
#include <unistd.h>
#include <omp.h>

#include "sha256.h"

////////////////////////   Block   /////////////////////

typedef struct __align__(4) _block
{
    unsigned int version;
    unsigned char prevhash[32];
    unsigned char merkle_root[32];
    unsigned int ntime;
    unsigned int nbits;
    unsigned int nonce;
} HashBlock;

////////////////////////   Utils   ///////////////////////

unsigned char decode(unsigned char c)
{
    switch(c)
    {
        case 'a': return 0x0a;
        case 'b': return 0x0b;
        case 'c': return 0x0c;
        case 'd': return 0x0d;
        case 'e': return 0x0e;
        case 'f': return 0x0f;
        case '0' ... '9': return c-'0';
    }
    return 0;
}

void convert_string_to_little_endian_bytes(unsigned char* out, char *in, size_t string_len)
{
    assert(string_len % 2 == 0);
    for(size_t s = 0, b = string_len/2-1; s < string_len; s+=2, --b)
    {
        out[b] = (unsigned char)(decode(in[s])<<4) + decode(in[s+1]);
    }
}

int little_endian_bit_comparison(const unsigned char *a, const unsigned char *b, size_t byte_len)
{
    for(int i=byte_len-1;i>=0;--i)
    {
        if(a[i] < b[i]) return -1;
        else if(a[i] > b[i]) return 1;
    }
    return 0;
}

void getline(char *str, size_t len, FILE *fp)
{
    int i=0;
    while( i<len && (str[i] = fgetc(fp)) != EOF && str[i++] != '\n');
    str[len-1] = '\0';
}

////////////////////////   Hash   ///////////////////////

void double_sha256(SHA256 *sha256_ctx, unsigned char *bytes, size_t len)
{
    SHA256 tmp;
    sha256(&tmp, (BYTE*)bytes, len);
    sha256(sha256_ctx, (BYTE*)&tmp, sizeof(tmp));
}

// Host-side SHA-256 transform for midstate computation
void host_sha256_transform(WORD state[8], const WORD data[16])
{
    const WORD k[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
    };
    
    WORD w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = data[i];
    }
    
    for (int i = 16; i < 64; ++i) {
        WORD s0 = ((w[i-15] >> 7) | (w[i-15] << 25)) ^ ((w[i-15] >> 18) | (w[i-15] << 14)) ^ (w[i-15] >> 3);
        WORD s1 = ((w[i-2] >> 17) | (w[i-2] << 15)) ^ ((w[i-2] >> 19) | (w[i-2] << 13)) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    
    WORD a = state[0], b = state[1], c = state[2], d = state[3];
    WORD e = state[4], f = state[5], g = state[6], h = state[7];
    
    for (int i = 0; i < 64; ++i) {
        WORD S1 = ((e >> 6) | (e << 26)) ^ ((e >> 11) | (e << 21)) ^ ((e >> 25) | (e << 7));
        WORD ch = (e & f) ^ ((~e) & g);
        WORD temp1 = h + S1 + ch + k[i] + w[i];
        WORD S0 = ((a >> 2) | (a << 30)) ^ ((a >> 13) | (a << 19)) ^ ((a >> 22) | (a << 10));
        WORD maj = (a & b) ^ (a & c) ^ (b & c);
        WORD temp2 = S0 + maj;
        
        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }
    
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

// Compute midstate and chunk1 base from block header
void compute_midstate_and_chunk1(const HashBlock &block, WORD midstate[8], WORD chunk1_base[16])
{
    const unsigned char *bytes = reinterpret_cast<const unsigned char*>(&block);
    
    // First 64 bytes -> chunk0 (16 words)
    WORD chunk0[16];
    for (int i = 0; i < 16; ++i) {
        int j = i * 4;
        chunk0[i] = (WORD(bytes[j]) << 24) | (WORD(bytes[j+1]) << 16) |
                    (WORD(bytes[j+2]) << 8) | WORD(bytes[j+3]);
    }
    
    // Compute midstate = SHA256_Transform(IV, chunk0)
    midstate[0] = 0x6a09e667;
    midstate[1] = 0xbb67ae85;
    midstate[2] = 0x3c6ef372;
    midstate[3] = 0xa54ff53a;
    midstate[4] = 0x510e527f;
    midstate[5] = 0x9b05688c;
    midstate[6] = 0x1f83d9ab;
    midstate[7] = 0x5be0cd19;
    host_sha256_transform(midstate, chunk0);
    
    // Next 16 bytes: merkle tail (4), ntime (4), nbits (4), nonce (4)
    const unsigned char *tail = bytes + 64;
    chunk1_base[0] = (WORD(tail[0]) << 24) | (WORD(tail[1]) << 16) |
                     (WORD(tail[2]) << 8) | WORD(tail[3]);
    chunk1_base[1] = (WORD(tail[4]) << 24) | (WORD(tail[5]) << 16) |
                     (WORD(tail[6]) << 8) | WORD(tail[7]);
    chunk1_base[2] = (WORD(tail[8]) << 24) | (WORD(tail[9]) << 16) |
                     (WORD(tail[10]) << 8) | WORD(tail[11]);
    chunk1_base[3] = 0; // placeholder for nonce (set per-thread)
    chunk1_base[4] = 0x80000000; // padding
    for (int i = 5; i < 15; ++i) {
        chunk1_base[i] = 0;
    }
    chunk1_base[15] = 80 * 8; // message length in bits (640)
}

////////////////////////   Device Code   /////////////////////

__constant__ WORD d_k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

__constant__ WORD d_midstate[8];
__constant__ WORD d_chunk1_base[16];

#define _d_rotr(v, s) ((v)>>(s) | (v)<<(32-(s)))
#define _d_swap(x, y) (((x)^=(y)), ((y)^=(x)), ((x)^=(y)))

__device__ void device_sha256_transform(SHA256 *ctx, const BYTE *msg)
{
    WORD a, b, c, d, e, f, g, h, i, j;
    
    WORD w[64];
    for(i=0, j=0;i<16;++i, j+=4)
        w[i] = (msg[j]<<24) | (msg[j+1]<<16) | (msg[j+2]<<8) | msg[j+3];
    
    for(i=16;i<64;++i) {
        WORD s0 = _d_rotr(w[i-15], 7) ^ _d_rotr(w[i-15], 18) ^ (w[i-15]>>3);
        WORD s1 = _d_rotr(w[i-2], 17) ^ _d_rotr(w[i-2], 19) ^ (w[i-2]>>10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    
    a = ctx->h[0]; b = ctx->h[1]; c = ctx->h[2]; d = ctx->h[3];
    e = ctx->h[4]; f = ctx->h[5]; g = ctx->h[6]; h = ctx->h[7];
    
    for(i=0;i<64;++i) {
        WORD S0 = _d_rotr(a, 2) ^ _d_rotr(a, 13) ^ _d_rotr(a, 22);
        WORD S1 = _d_rotr(e, 6) ^ _d_rotr(e, 11) ^ _d_rotr(e, 25);
        WORD ch = (e & f) ^ ((~e) & g);
        WORD maj = (a & b) ^ (a & c) ^ (b & c);
        WORD temp1 = h + S1 + ch + d_k[i] + w[i];
        WORD temp2 = S0 + maj;
        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }
    
    ctx->h[0] += a; ctx->h[1] += b; ctx->h[2] += c; ctx->h[3] += d;
    ctx->h[4] += e; ctx->h[5] += f; ctx->h[6] += g; ctx->h[7] += h;
}

__device__ void device_sha256(SHA256 *ctx, const BYTE *msg, size_t len)
{
    ctx->h[0] = 0x6a09e667; ctx->h[1] = 0xbb67ae85;
    ctx->h[2] = 0x3c6ef372; ctx->h[3] = 0xa54ff53a;
    ctx->h[4] = 0x510e527f; ctx->h[5] = 0x9b05688c;
    ctx->h[6] = 0x1f83d9ab; ctx->h[7] = 0x5be0cd19;
    
    size_t remain = len % 64;
    size_t total_len = len - remain;
    
    for(WORD i=0;i<total_len;i+=64)
        device_sha256_transform(ctx, &msg[i]);
    
    BYTE m[64] = {};
    WORD i, j;
    for(i=total_len, j=0;i<len;++i, ++j)
        m[j] = msg[i];
    m[j++] = 0x80;
    
    if(j > 56) {
        device_sha256_transform(ctx, m);
        for(i=0;i<64;++i) m[i] = 0;
    }
    
    unsigned long long L = len * 8;
    m[63] = L; m[62] = L >> 8; m[61] = L >> 16; m[60] = L >> 24;
    m[59] = L >> 32; m[58] = L >> 40; m[57] = L >> 48; m[56] = L >> 56;
    device_sha256_transform(ctx, m);
    
    for(i=0;i<32;i+=4) {
        _d_swap(ctx->b[i], ctx->b[i+3]);
        _d_swap(ctx->b[i+1], ctx->b[i+2]);
    }
}

__device__ __forceinline__ WORD _d_sigma0(WORD x) {
    return _d_rotr(x, 7) ^ _d_rotr(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ WORD _d_sigma1(WORD x) {
    return _d_rotr(x, 17) ^ _d_rotr(x, 19) ^ (x >> 10);
}

__device__ __forceinline__ WORD _d_SIGMA0(WORD x) {
    return _d_rotr(x, 2) ^ _d_rotr(x, 13) ^ _d_rotr(x, 22);
}

__device__ __forceinline__ WORD _d_SIGMA1(WORD x) {
    return _d_rotr(x, 6) ^ _d_rotr(x, 11) ^ _d_rotr(x, 25);
}

__device__ __forceinline__ WORD _d_choose(WORD x, WORD y, WORD z) {
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ WORD _d_majority(WORD x, WORD y, WORD z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ void device_sha256_transform_words(WORD state[8], const WORD data[16])
{
    WORD w[64];
    
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        w[i] = data[i];
    }
    
    #pragma unroll
    for (int i = 16; i < 64; ++i) {
        w[i] = _d_sigma1(w[i-2]) + w[i-7] + _d_sigma0(w[i-15]) + w[i-16];
    }
    
    WORD a = state[0], b = state[1], c = state[2], d = state[3];
    WORD e = state[4], f = state[5], g = state[6], h = state[7];
    
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        WORD temp1 = h + _d_SIGMA1(e) + _d_choose(e, f, g) + d_k[i] + w[i];
        WORD temp2 = _d_SIGMA0(a) + _d_majority(a, b, c);
        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }
    
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

// Optimized: Start from midstate, only process chunk1 + final hash
__device__ void device_double_sha256_from_midstate(SHA256 *sha256_ctx, unsigned int nonce)
{
    // Copy midstate (already processed chunk0)
    WORD state[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        state[i] = d_midstate[i];
    }
    
    // Prepare chunk1 with nonce
    WORD chunk1[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        chunk1[i] = d_chunk1_base[i];
    }
    
    // Insert nonce (little-endian to big-endian conversion)
    chunk1[3] = ((nonce & 0x000000FFu) << 24) |
                ((nonce & 0x0000FF00u) << 8)  |
                ((nonce & 0x00FF0000u) >> 8)  |
                ((nonce & 0xFF000000u) >> 24);
    
    // Process chunk1
    device_sha256_transform_words(state, chunk1);
    
    // Convert state to bytes for second hash
    unsigned char intermediate[32];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        WORD val = state[i];
        intermediate[4*i + 0] = (val >> 24) & 0xff;
        intermediate[4*i + 1] = (val >> 16) & 0xff;
        intermediate[4*i + 2] = (val >> 8) & 0xff;
        intermediate[4*i + 3] = val & 0xff;
    }
    
    // Second SHA-256
    device_sha256(sha256_ctx, intermediate, 32);
}

__device__ inline int device_little_endian_bit_comparison(const unsigned char *a, const unsigned char *b, size_t byte_len)
{
    for(int i=byte_len-1; i>=0; --i) {
        if(a[i] < b[i]) return -1;
        else if(a[i] > b[i]) return 1;
    }
    return 0;
}

////////////////////   Merkle Root   /////////////////////

void calc_merkle_root(unsigned char *root, int count, char **branch)
{
    size_t total_count = count;
    unsigned char *raw_list = new unsigned char[(total_count+1)*32];
    unsigned char **list = new unsigned char*[total_count+1];

    for(int i=0;i<total_count; ++i) {
        list[i] = raw_list + i * 32;
        convert_string_to_little_endian_bytes(list[i], branch[i], 64);
    }
    list[total_count] = raw_list + total_count*32;

    while(total_count > 1) {
        if(total_count % 2 == 1)
            memcpy(list[total_count], list[total_count-1], 32);

        int i, j;
        for(i=0, j=0;i<total_count;i+=2, ++j)
            double_sha256((SHA256*)list[j], list[i], 64);

        total_count = j;
    }

    memcpy(root, list[0], 32);
    delete[] raw_list;
    delete[] list;
}

////////////////////////   Mining Kernel   /////////////////////

// Device-side work structure for persistent kernel
struct DeviceWork {
    unsigned long long next_nonce;
    unsigned long long end_nonce;
    int found;
    unsigned int result_nonce;
};

// Persistent kernel: continuously fetch work until exhausted or found
__global__ void persistent_mine_kernel(
    unsigned char *target,
    DeviceWork *work,
    unsigned int batch_size
)
{
    __shared__ unsigned long long s_batch_start;
    __shared__ int s_local_found;
    
    while (true) {
        // Check global found flag
        if (work->found) {
            return;
        }
        
        // Thread 0 fetches next batch
        if (threadIdx.x == 0) {
            s_batch_start = atomicAdd(
                reinterpret_cast<unsigned long long*>(&work->next_nonce),
                static_cast<unsigned long long>(batch_size)
            );
            s_local_found = 0;
        }
        __syncthreads();
        
        unsigned long long batch_start = s_batch_start;
        
        // Check if we've exhausted all nonces
        if (batch_start >= work->end_nonce) {
            return;
        }
        
        unsigned long long batch_end = batch_start + batch_size;
        if (batch_end > work->end_nonce) {
            batch_end = work->end_nonce;
        }
        
        // Each thread processes multiple nonces with stride
        for (unsigned long long nonce_val = batch_start + threadIdx.x;
             nonce_val < batch_end && !s_local_found;
             nonce_val += blockDim.x)
        {
            // Periodic check for early exit
            if (((nonce_val - batch_start) & 0x1Fu) == 0 && work->found) {
                s_local_found = 1;
                break;
            }
            
            unsigned int nonce = static_cast<unsigned int>(nonce_val);
            
            SHA256 hash;
            device_double_sha256_from_midstate(&hash, nonce);
            
            if (device_little_endian_bit_comparison(hash.b, target, 32) < 0) {
                if (atomicCAS(&work->found, 0, 1) == 0) {
                    work->result_nonce = nonce;
                    __threadfence_system();
                }
                s_local_found = 1;
                break;
            }
        }
        
        __syncthreads();
        
        // If found, exit immediately
        if (s_local_found || work->found) {
            return;
        }
    }
}

// Chunked kernel using global memory for midstate (for multi-stream overlap)
__global__ void mine_kernel_global(
    unsigned char *target,
    WORD *midstate,
    WORD *chunk1_base,
    unsigned int start_nonce,
    unsigned int nonce_count,
    int *found,
    unsigned int *result_nonce
)
{
    __shared__ WORD s_midstate[8];
    __shared__ WORD s_chunk1_base[16];
    __shared__ unsigned char s_target[32];
    
    // Load midstate, chunk1_base, and target into shared memory once
    if (threadIdx.x < 8) {
        s_midstate[threadIdx.x] = midstate[threadIdx.x];
    }
    if (threadIdx.x < 16) {
        s_chunk1_base[threadIdx.x] = chunk1_base[threadIdx.x];
    }
    if (threadIdx.x < 32) {
        s_target[threadIdx.x] = target[threadIdx.x];
    }
    __syncthreads();
    
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    
    for (unsigned int i = idx; i < nonce_count; i += stride) {
        // Check if already found every 32 iterations
        if ((i & 0x1F) == 0 && *found) {
            return;
        }
        
        unsigned int nonce = start_nonce + i;
        
        // Compute hash from midstate
        WORD state[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            state[j] = s_midstate[j];
        }
        
        WORD chunk1[16];
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            chunk1[j] = s_chunk1_base[j];
        }
        
        // Use hardware byte permutation for endianness conversion (1 instruction vs 11)
        chunk1[3] = __byte_perm(nonce, 0, 0x0123);
        
        device_sha256_transform_words(state, chunk1);
        
        unsigned char intermediate[32];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            WORD val = state[j];
            intermediate[4*j + 0] = (val >> 24) & 0xff;
            intermediate[4*j + 1] = (val >> 16) & 0xff;
            intermediate[4*j + 2] = (val >> 8) & 0xff;
            intermediate[4*j + 3] = val & 0xff;
        }
        
        SHA256 hash;
        device_sha256(&hash, intermediate, 32);
        
        // Use shared memory target for faster comparison
        if (device_little_endian_bit_comparison(hash.b, s_target, 32) < 0) {
            if (atomicCAS(found, 0, 1) == 0) {
                *result_nonce = nonce;
            }
            return;
        }
    }
}

////////////////////////   Host Mining   /////////////////////

// Persistent kernel mining (optimized for single block)
bool mine_block_persistent(HashBlock &block, unsigned char target[32])
{
    const unsigned long long total_nonces = 0x100000000ULL;
    const int threads_per_block = 256;
    const int blocks_per_grid = 512;
    const unsigned int batch_size = 262144;  // 256K nonces per atomic fetch (4x increase for less contention)
    
    // Compute midstate on host
    WORD midstate[8];
    WORD chunk1_base[16];
    compute_midstate_and_chunk1(block, midstate, chunk1_base);
    
    // Copy to device constant memory
    cudaMemcpyToSymbol(d_midstate, midstate, sizeof(midstate));
    cudaMemcpyToSymbol(d_chunk1_base, chunk1_base, sizeof(chunk1_base));
    
    // Allocate device memory
    unsigned char *d_target;
    DeviceWork *d_work;
    cudaMalloc(&d_target, 32);
    cudaMalloc(&d_work, sizeof(DeviceWork));
    
    cudaMemcpy(d_target, target, 32, cudaMemcpyHostToDevice);
    
    // Initialize work structure
    DeviceWork h_work;
    h_work.next_nonce = 0ULL;
    h_work.end_nonce = total_nonces;
    h_work.found = 0;
    h_work.result_nonce = 0;
    cudaMemcpy(d_work, &h_work, sizeof(DeviceWork), cudaMemcpyHostToDevice);
    
    // Launch persistent kernel (single launch for entire search space)
    persistent_mine_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_target,
        d_work,
        batch_size
    );
    
    // Wait for completion
    cudaDeviceSynchronize();
    
    // Retrieve results
    cudaMemcpy(&h_work, d_work, sizeof(DeviceWork), cudaMemcpyDeviceToHost);
    
    cudaFree(d_target);
    cudaFree(d_work);
    
    if (h_work.found) {
        block.nonce = h_work.result_nonce;
        return true;
    }
    
    block.nonce = 0;
    return false;
}


////////////////////////   Multi-Stream Pipeline   /////////////////////

struct BlockTask {
    HashBlock block;
    unsigned char target[32];
    WORD midstate[8];
    WORD chunk1_base[16];
    bool completed;
    bool found;
};

// Stream pool context (persistent across multiple mine_blocks_pipeline calls)
struct StreamPoolContext {
    // CUDA resources (persistent, allocated once)
    cudaStream_t stream;
    cudaEvent_t event;
    unsigned char *d_target;
    WORD *d_midstate;
    WORD *d_chunk1_base;
    int *h_found_pinned;
    unsigned int *h_result_nonce_pinned;
    int *d_found;
    unsigned int *d_result_nonce;
    
    // Task state (reset per call)
    int block_idx;
    unsigned int current_nonce;
    bool busy;
    
    // Pool management
    bool initialized;
};

// Global static stream pool (allocated once, reused forever)
static StreamPoolContext g_stream_pool[4];
static bool g_pool_initialized = false;

// Initialize stream pool (called once on first use)
static void initialize_stream_pool()
{
    if (g_pool_initialized) return;
    
    for (int i = 0; i < 4; ++i) {
        StreamPoolContext &ctx = g_stream_pool[i];
        
        // Create persistent CUDA resources
        cudaStreamCreateWithFlags(&ctx.stream, cudaStreamNonBlocking);
        cudaEventCreate(&ctx.event);
        
        // Allocate device memory (never freed until program exit)
        cudaMalloc(&ctx.d_target, 32);
        cudaMalloc(&ctx.d_midstate, 8 * sizeof(WORD));
        cudaMalloc(&ctx.d_chunk1_base, 16 * sizeof(WORD));
        
        // Allocate pinned host memory for zero-copy
        cudaHostAlloc(&ctx.h_found_pinned, sizeof(int), cudaHostAllocMapped);
        cudaHostAlloc(&ctx.h_result_nonce_pinned, sizeof(unsigned int), cudaHostAllocMapped);
        
        // Get device pointers
        cudaHostGetDevicePointer(&ctx.d_found, ctx.h_found_pinned, 0);
        cudaHostGetDevicePointer(&ctx.d_result_nonce, ctx.h_result_nonce_pinned, 0);
        
        // Mark as initialized
        ctx.initialized = true;
        ctx.block_idx = -1;
        ctx.current_nonce = 0;
        ctx.busy = false;
    }
    
    g_pool_initialized = true;
}

// Reset stream context for new task (reuse existing resources)
static void reset_stream_context(StreamPoolContext &ctx)
{
    ctx.block_idx = -1;
    ctx.current_nonce = 0;
    ctx.busy = false;
    *ctx.h_found_pinned = 0;
    *ctx.h_result_nonce_pinned = 0;
}

bool mine_blocks_pipeline(std::vector<BlockTask> &tasks)
{
    if (tasks.empty()) return true;
    
    const int max_streams = std::min(4, static_cast<int>(tasks.size()));
    const int threads_per_block = 256;
    const int blocks_per_grid = 512;
    const unsigned int chunk_size = 16777216;  // 16M nonces per kernel
    const unsigned long long total_nonces = 0x100000000ULL;
    
    // Initialize stream pool on first use (lazy initialization)
    initialize_stream_pool();
    
    // Reset contexts for new tasks (reuse pool, no allocation/deallocation)
    for (int i = 0; i < max_streams; ++i) {
        reset_stream_context(g_stream_pool[i]);
    }
    
    int next_task = 0;
    int completed_count = 0;
    
    // Track which streams just completed a kernel (to avoid redundant queries)
    bool kernel_completed[4] = {false, false, false, false};
    
    while (completed_count < tasks.size()) {
        // Reset completion flags for this iteration
        for (int i = 0; i < max_streams; ++i) {
            kernel_completed[i] = false;
        }
        
        // Step 1: Check for completed chunks and handle results
        for (int i = 0; i < max_streams; ++i) {
            if (g_stream_pool[i].busy) {
                cudaError_t status = cudaEventQuery(g_stream_pool[i].event);
                
                if (status == cudaSuccess) {
                    StreamPoolContext &ctx = g_stream_pool[i];
                    kernel_completed[i] = true;  // Mark as just completed
                    
                    // Direct read from pinned host memory (zero-copy, no sync needed!)
                    // The kernel writes to device pointer which maps to host pinned memory
                    int found_value = *ctx.h_found_pinned;
                    unsigned int result_nonce_value = *ctx.h_result_nonce_pinned;
                    
                    // If found or exhausted all nonces, complete task
                    if (found_value || ctx.current_nonce >= total_nonces) {
                        BlockTask &task = tasks[ctx.block_idx];
                        task.completed = true;
                        if (found_value) {
                            task.found = true;
                            task.block.nonce = result_nonce_value;
                        } else {
                            task.found = false;
                            task.block.nonce = 0;
                        }
                        
                        ctx.busy = false;
                        ctx.block_idx = -1;
                        ++completed_count;
                    }
                }
            }
        }
        
        // Step 2: Launch new work on available streams
        for (int i = 0; i < max_streams; ++i) {
            if (!g_stream_pool[i].busy && next_task < tasks.size()) {
                BlockTask &task = tasks[next_task];
                StreamPoolContext &ctx = g_stream_pool[i];
                
                // Initialize new task
                ctx.block_idx = next_task;
                ctx.current_nonce = 0;
                
                // Initialize pinned memory to 0 (direct host write)
                *ctx.h_found_pinned = 0;
                *ctx.h_result_nonce_pinned = 0;
                
                // Copy midstate, chunk1_base, target to device
                cudaMemcpyAsync(ctx.d_midstate, task.midstate, 
                               8 * sizeof(WORD),
                               cudaMemcpyHostToDevice, ctx.stream);
                cudaMemcpyAsync(ctx.d_chunk1_base, task.chunk1_base, 
                               16 * sizeof(WORD),
                               cudaMemcpyHostToDevice, ctx.stream);
                cudaMemcpyAsync(ctx.d_target, task.target, 32,
                               cudaMemcpyHostToDevice, ctx.stream);
                
                ctx.busy = true;
                ++next_task;
            }
        }
        
        // Step 3: Launch next chunk on busy streams
        for (int i = 0; i < max_streams; ++i) {
            if (g_stream_pool[i].busy && !(*g_stream_pool[i].h_found_pinned) && 
                g_stream_pool[i].current_nonce < total_nonces) {
                
                StreamPoolContext &ctx = g_stream_pool[i];
                
                // Launch next chunk if: first launch OR previous kernel just completed
                // This avoids redundant cudaEventQuery since we already checked in Step 1
                if (ctx.current_nonce == 0 || kernel_completed[i]) {
                    unsigned int nonces_to_process = chunk_size;
                    if (ctx.current_nonce + nonces_to_process > total_nonces) {
                        nonces_to_process = total_nonces - ctx.current_nonce;
                    }
                    
                    // Launch chunk kernel
                    mine_kernel_global<<<blocks_per_grid, threads_per_block, 0, ctx.stream>>>(
                        ctx.d_target,
                        ctx.d_midstate,
                        ctx.d_chunk1_base,
                        ctx.current_nonce,
                        nonces_to_process,
                        ctx.d_found,
                        ctx.d_result_nonce
                    );
                    
                    cudaEventRecord(ctx.event, ctx.stream);
                    ctx.current_nonce += nonces_to_process;
                }
            }
        }
        
        // Increased sleep to reduce CPU busy-polling
        // 50μs instead of 10μs reduces loop iterations by 80%
        usleep(50);
    }
    
    // No cleanup needed! Stream pool is persistent and reused
    // Resources will be automatically freed when program exits
    
    return true;
}

////////////////////////   Main   /////////////////////

void solve(FILE *fin, FILE *fout)
{
    char version[9], prevhash[65], ntime[9], nbits[9];
    int tx;
    char *raw_merkle_branch;
    char **merkle_branch;

    getline(version, 9, fin);
    getline(prevhash, 65, fin);
    getline(ntime, 9, fin);
    getline(nbits, 9, fin);
    fscanf(fin, "%d\n", &tx);

    raw_merkle_branch = new char [static_cast<size_t>(tx) * 65];
    merkle_branch = new char *[tx];
    for(int i=0;i<tx;++i) {
        merkle_branch[i] = raw_merkle_branch + static_cast<size_t>(i) * 65;
        getline(merkle_branch[i], 65, fin);
        merkle_branch[i][64] = '\0';
    }

    unsigned char merkle_root[32];
    calc_merkle_root(merkle_root, tx, merkle_branch);

    HashBlock block;
    convert_string_to_little_endian_bytes((unsigned char *)&block.version, version, 8);
    convert_string_to_little_endian_bytes(block.prevhash, prevhash, 64);
    memcpy(block.merkle_root, merkle_root, 32);
    convert_string_to_little_endian_bytes((unsigned char *)&block.nbits, nbits, 8);
    convert_string_to_little_endian_bytes((unsigned char *)&block.ntime, ntime, 8);
    block.nonce = 0;

    unsigned int exp = block.nbits >> 24;
    unsigned int mant = block.nbits & 0xffffff;
    unsigned char target_hex[32] = {};

    unsigned int shift = 8 * (exp - 3);
    unsigned int sb = shift / 8;
    unsigned int rb = shift % 8;

    target_hex[sb    ] = static_cast<unsigned char>(mant << rb);
    target_hex[sb + 1] = static_cast<unsigned char>(mant >> (8-rb));
    target_hex[sb + 2] = static_cast<unsigned char>(mant >> (16-rb));
    target_hex[sb + 3] = static_cast<unsigned char>(mant >> (24-rb));

    bool solved = mine_block_persistent(block, target_hex);

    // Verify on host
    SHA256 sha256_ctx;
    double_sha256(&sha256_ctx, (unsigned char*)&block, 80);

    if(!(solved && little_endian_bit_comparison(sha256_ctx.b, target_hex, 32) < 0)) {
        block.nonce = 0;
    }

    for(int i=0;i<4;++i) {
        fprintf(fout, "%02x", ((unsigned char*)&block.nonce)[i]);
    }
    fprintf(fout, "\n");

    delete[] merkle_branch;
    delete[] raw_merkle_branch;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: hw4 <in> <out>\n");
        return 1;
    }

    FILE *fin = fopen(argv[1], "r");
    FILE *fout = fopen(argv[2], "w");
    if (!fin || !fout) {
        fprintf(stderr, "Error opening files\n");
        if (fin) fclose(fin);
        if (fout) fclose(fout);
        return 1;
    }

    int totalblock;
    if (fscanf(fin, "%d\n", &totalblock) != 1 || totalblock < 0) {
        fclose(fin);
        fclose(fout);
        return 1;
    }
    fprintf(fout, "%d\n", totalblock);

    // Use pipeline for multiple blocks, single-stream for single block
    if (totalblock > 1) {
        // Multi-block: use pipeline
        std::vector<BlockTask> tasks(totalblock);
        
        // Phase 1: Serial file reading and block construction
        for (int i = 0; i < totalblock; ++i) {
            char version[9], prevhash[65], ntime[9], nbits[9];
            int tx;
            char *raw_merkle_branch;
            char **merkle_branch;

            getline(version, 9, fin);
            getline(prevhash, 65, fin);
            getline(ntime, 9, fin);
            getline(nbits, 9, fin);
            fscanf(fin, "%d\n", &tx);

            raw_merkle_branch = new char [static_cast<size_t>(tx) * 65];
            merkle_branch = new char *[tx];
            for(int j=0; j<tx; ++j) {
                merkle_branch[j] = raw_merkle_branch + static_cast<size_t>(j) * 65;
                getline(merkle_branch[j], 65, fin);
                merkle_branch[j][64] = '\0';
            }

            unsigned char merkle_root[32];
            calc_merkle_root(merkle_root, tx, merkle_branch);

            convert_string_to_little_endian_bytes((unsigned char *)&tasks[i].block.version, version, 8);
            convert_string_to_little_endian_bytes(tasks[i].block.prevhash, prevhash, 64);
            memcpy(tasks[i].block.merkle_root, merkle_root, 32);
            convert_string_to_little_endian_bytes((unsigned char *)&tasks[i].block.nbits, nbits, 8);
            convert_string_to_little_endian_bytes((unsigned char *)&tasks[i].block.ntime, ntime, 8);
            tasks[i].block.nonce = 0;

            unsigned int exp = tasks[i].block.nbits >> 24;
            unsigned int mant = tasks[i].block.nbits & 0xffffff;
            memset(tasks[i].target, 0, 32);

            unsigned int shift = 8 * (exp - 3);
            unsigned int sb = shift / 8;
            unsigned int rb = shift % 8;

            tasks[i].target[sb    ] = static_cast<unsigned char>(mant << rb);
            tasks[i].target[sb + 1] = static_cast<unsigned char>(mant >> (8-rb));
            tasks[i].target[sb + 2] = static_cast<unsigned char>(mant >> (16-rb));
            tasks[i].target[sb + 3] = static_cast<unsigned char>(mant >> (24-rb));
            
            tasks[i].completed = false;
            tasks[i].found = false;

            delete[] merkle_branch;
            delete[] raw_merkle_branch;
        }
        
        // Phase 2: Parallel midstate computation (CPU-bound SHA-256)
        // Use OpenMP to parallelize across all blocks
        // With 72 CPU threads available, 4 blocks will be computed in parallel
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < totalblock; ++i) {
            compute_midstate_and_chunk1(tasks[i].block, tasks[i].midstate, tasks[i].chunk1_base);
        }
        
        // Run pipeline
        bool success = mine_blocks_pipeline(tasks);
        
        // Write results
        for (int i = 0; i < totalblock; ++i) {
            // Verify on host
            SHA256 sha256_ctx;
            double_sha256(&sha256_ctx, (unsigned char*)&tasks[i].block, 80);
            
            if(!(tasks[i].found && little_endian_bit_comparison(sha256_ctx.b, tasks[i].target, 32) < 0)) {
                tasks[i].block.nonce = 0;
            }
            
            for(int j=0; j<4; ++j) {
                fprintf(fout, "%02x", ((unsigned char*)&tasks[i].block.nonce)[j]);
            }
            fprintf(fout, "\n");
        }
    } else {
        // Single block: use original path
        for (int i = 0; i < totalblock; ++i) {
            solve(fin, fout);
        }
    }

    fclose(fin);
    fclose(fout);
    return 0;
}
