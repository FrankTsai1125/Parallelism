//***********************************************************************************
// CUDA Bitcoin Miner - Parallel Implementation
// Modified from sequential version for GPU acceleration
//***********************************************************************************

#include <cstdio>
#include <cstring>

#include <cassert>

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

//convert one hex-codec char to binary
unsigned char decode(unsigned char c)
{
    switch(c)
    {
        case 'a':
            return 0x0a;
        case 'b':
            return 0x0b;
        case 'c':
            return 0x0c;
        case 'd':
            return 0x0d;
        case 'e':
            return 0x0e;
        case 'f':
            return 0x0f;
        case '0' ... '9':
            return c-'0';
    }
    return 0;
}


// convert hex string to binary
//
// in: input string
// string_len: the length of the input string
//      '\0' is not included in string_len!!!
// out: output bytes array
void convert_string_to_little_endian_bytes(unsigned char* out, char *in, size_t string_len)
{
    assert(string_len % 2 == 0);

    size_t s = 0;
    size_t b = string_len/2-1;

    for(s = 0, b = string_len/2-1; s < string_len; s+=2, --b)
    {
        out[b] = (unsigned char)(decode(in[s])<<4) + decode(in[s+1]);
    }
}

int little_endian_bit_comparison(const unsigned char *a, const unsigned char *b, size_t byte_len)
{
    // compared from lowest bit
    for(int i=byte_len-1;i>=0;--i)
    {
        if(a[i] < b[i])
            return -1;
        else if(a[i] > b[i])
            return 1;
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

void prepare_sha256_chunks(const HashBlock &block, WORD chunk0[16], WORD chunk1_base[16])
{
    const unsigned char *bytes = reinterpret_cast<const unsigned char*>(&block);
    
    // First 64 bytes -> chunk0 (16 words)
    for (int i = 0; i < 16; ++i) {
        int j = i * 4;
        chunk0[i] = (WORD(bytes[j]) << 24) | (WORD(bytes[j+1]) << 16) |
                    (WORD(bytes[j+2]) << 8) | WORD(bytes[j+3]);
    }
    
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

// Device SHA-256 constants
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

// Precomputed message chunks in constant memory (used by optimized SHA-256 pipeline)
__constant__ WORD d_chunk0_words[16];
__constant__ WORD d_chunk1_words_base[16];

#define _d_rotr(v, s) ((v)>>(s) | (v)<<(32-(s)))
#define _d_swap(x, y) (((x)^=(y)), ((y)^=(x)), ((x)^=(y)))

__device__ void device_sha256_transform(SHA256 *ctx, const BYTE *msg)
{
	WORD a, b, c, d, e, f, g, h;
	WORD i, j;
	
	WORD w[64];
	for(i=0, j=0;i<16;++i, j+=4)
	{
		w[i] = (msg[j]<<24) | (msg[j+1]<<16) | (msg[j+2]<<8) | (msg[j+3]);
	}
	
	for(i=16;i<64;++i)
	{
		WORD s0 = (_d_rotr(w[i-15], 7)) ^ (_d_rotr(w[i-15], 18)) ^ (w[i-15]>>3);
		WORD s1 = (_d_rotr(w[i-2], 17)) ^ (_d_rotr(w[i-2], 19))  ^ (w[i-2]>>10);
		w[i] = w[i-16] + s0 + w[i-7] + s1;
	}
	
	a = ctx->h[0];
	b = ctx->h[1];
	c = ctx->h[2];
	d = ctx->h[3];
	e = ctx->h[4];
	f = ctx->h[5];
	g = ctx->h[6];
	h = ctx->h[7];
	
	for(i=0;i<64;++i)
	{
		WORD S0 = (_d_rotr(a, 2)) ^ (_d_rotr(a, 13)) ^ (_d_rotr(a, 22));
		WORD S1 = (_d_rotr(e, 6)) ^ (_d_rotr(e, 11)) ^ (_d_rotr(e, 25));
		WORD ch = (e & f) ^ ((~e) & g);
		WORD maj = (a & b) ^ (a & c) ^ (b & c);
		WORD temp1 = h + S1 + ch + d_k[i] + w[i];
		WORD temp2 = S0 + maj;
		
		h = g;
		g = f;
		f = e;
		e = d + temp1;
		d = c;
		c = b;
		b = a;
		a = temp1 + temp2;
	}
	
	ctx->h[0] += a;
	ctx->h[1] += b;
	ctx->h[2] += c;
	ctx->h[3] += d;
	ctx->h[4] += e;
	ctx->h[5] += f;
	ctx->h[6] += g;
	ctx->h[7] += h;
}

__device__ void device_sha256(SHA256 *ctx, const BYTE *msg, size_t len)
{
	ctx->h[0] = 0x6a09e667;
	ctx->h[1] = 0xbb67ae85;
	ctx->h[2] = 0x3c6ef372;
	ctx->h[3] = 0xa54ff53a;
	ctx->h[4] = 0x510e527f;
	ctx->h[5] = 0x9b05688c;
	ctx->h[6] = 0x1f83d9ab;
	ctx->h[7] = 0x5be0cd19;
	
	WORD i, j;
	size_t remain = len % 64;
	size_t total_len = len - remain;
	
	for(i=0;i<total_len;i+=64)
	{
		device_sha256_transform(ctx, &msg[i]);
	}
	
	BYTE m[64] = {};
	for(i=total_len, j=0;i<len;++i, ++j)
	{
		m[j] = msg[i];
	}
	
	m[j++] = 0x80;
	
	if(j > 56)
	{
		device_sha256_transform(ctx, m);
		for(i=0;i<64;++i) m[i] = 0;
	}
	
	unsigned long long L = len * 8;
	m[63] = L;
	m[62] = L >> 8;
	m[61] = L >> 16;
	m[60] = L >> 24;
	m[59] = L >> 32;
	m[58] = L >> 40;
	m[57] = L >> 48;
	m[56] = L >> 56;
	device_sha256_transform(ctx, m);
	
	for(i=0;i<32;i+=4)
	{
        _d_swap(ctx->b[i], ctx->b[i+3]);
        _d_swap(ctx->b[i+1], ctx->b[i+2]);
	}
}

__device__ __forceinline__ WORD _d_sigma0(WORD x)
{
    return _d_rotr(x, 7) ^ _d_rotr(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ WORD _d_sigma1(WORD x)
{
    return _d_rotr(x, 17) ^ _d_rotr(x, 19) ^ (x >> 10);
}

__device__ __forceinline__ WORD _d_SIGMA0(WORD x)
{
    return _d_rotr(x, 2) ^ _d_rotr(x, 13) ^ _d_rotr(x, 22);
}

__device__ __forceinline__ WORD _d_SIGMA1(WORD x)
{
    return _d_rotr(x, 6) ^ _d_rotr(x, 11) ^ _d_rotr(x, 25);
}

__device__ __forceinline__ WORD _d_choose(WORD x, WORD y, WORD z)
{
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ WORD _d_majority(WORD x, WORD y, WORD z)
{
    return (x & y) ^ (x & z) ^ (y & z);
}

// Optimized SHA-256 round operating directly on message words
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
    
    WORD a = state[0];
    WORD b = state[1];
    WORD c = state[2];
    WORD d = state[3];
    WORD e = state[4];
    WORD f = state[5];
    WORD g = state[6];
    WORD h = state[7];
    
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        WORD temp1 = h + _d_SIGMA1(e) + _d_choose(e, f, g) + d_k[i] + w[i];
        WORD temp2 = _d_SIGMA0(a) + _d_majority(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }
    
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

// Optimized double SHA-256 leveraging precomputed message chunks
__device__ void device_double_sha256_precomputed(SHA256 *sha256_ctx, unsigned int nonce)
{
    WORD state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    device_sha256_transform_words(state, d_chunk0_words);
    
    WORD chunk1[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        chunk1[i] = d_chunk1_words_base[i];
    }
    
    chunk1[3] = ((nonce & 0x000000FFu) << 24) |
                ((nonce & 0x0000FF00u) << 8)  |
                ((nonce & 0x00FF0000u) >> 8)  |
                ((nonce & 0xFF000000u) >> 24);
    
    device_sha256_transform_words(state, chunk1);
    
    unsigned char intermediate[32];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        WORD val = state[i];
        intermediate[4*i + 0] = (val >> 24) & 0xff;
        intermediate[4*i + 1] = (val >> 16) & 0xff;
        intermediate[4*i + 2] = (val >> 8) & 0xff;
        intermediate[4*i + 3] = val & 0xff;
    }
    
    device_sha256(sha256_ctx, intermediate, 32);
}

// Device version of double sha256 (original, kept for compatibility)
__device__ void device_double_sha256(SHA256 *sha256_ctx, unsigned char *bytes, size_t len)
{
    SHA256 tmp;
    device_sha256(&tmp, (BYTE*)bytes, len);
    device_sha256(sha256_ctx, (BYTE*)&tmp, sizeof(tmp));
}

// Device comparison function
__device__ inline int device_little_endian_bit_comparison(const unsigned char *a, const unsigned char *b, size_t byte_len)
{
    // Compare from lowest bit (highest index)
    for(int i=byte_len-1; i>=0; --i)
    {
        if(a[i] < b[i])
            return -1;
        else if(a[i] > b[i])
            return 1;
    }
    return 0;
}


////////////////////   Merkle Root   /////////////////////


// calculate merkle root from several merkle branches
// root: output hash will store here (little-endian)
// branch: merkle branch  (big-endian)
// count: total number of merkle branch
void calc_merkle_root(unsigned char *root, int count, char **branch)
{
    size_t total_count = count; // merkle branch
    unsigned char *raw_list = new unsigned char[(total_count+1)*32];
    unsigned char **list = new unsigned char*[total_count+1];

    // copy each branch to the list
    for(int i=0;i<total_count; ++i)
    {
        list[i] = raw_list + i * 32;
        //convert hex string to bytes array and store them into the list
        convert_string_to_little_endian_bytes(list[i], branch[i], 64);
    }

    list[total_count] = raw_list + total_count*32;


    // calculate merkle root
    while(total_count > 1)
    {
        
        // hash each pair
        int i, j;

        if(total_count % 2 == 1)  //odd, 
        {
            memcpy(list[total_count], list[total_count-1], 32);
        }

        for(i=0, j=0;i<total_count;i+=2, ++j)
        {
            // this part is slightly tricky,
            //   because of the implementation of the double_sha256,
            //   we can avoid the memory begin overwritten during our sha256d calculation
            // double_sha:
            //     tmp = hash(list[0]+list[1])
            //     list[0] = hash(tmp)
            double_sha256((SHA256*)list[j], list[i], 64);
        }

        total_count = j;
    }

    memcpy(root, list[0], 32);

    delete[] raw_list;
    delete[] list;
}


////////////////////////   CUDA Kernel   /////////////////////

struct DeviceWork {
    unsigned long long next_nonce;
    unsigned long long end_nonce;
    int found;
    unsigned int result;
};

__global__ void mine_bitcoin_kernel(
    unsigned char *target,
    DeviceWork *work,
    unsigned int batch_size
)
{
    __shared__ unsigned long long shared_base;
    __shared__ int block_found;

    while (true) {
        if (work->found) {
            return;
        }

        if (threadIdx.x == 0) {
            shared_base = atomicAdd(reinterpret_cast<unsigned long long*>(&work->next_nonce),
                                    static_cast<unsigned long long>(batch_size));
            block_found = 0;
        }
        __syncthreads();

        unsigned long long base = shared_base;
        if (base >= work->end_nonce) {
            return;
        }

        unsigned long long chunk_end = base + batch_size;
        if (chunk_end > work->end_nonce) {
            chunk_end = work->end_nonce;
        }

        for (unsigned long long nonce_idx = base + threadIdx.x;
             nonce_idx < chunk_end && !block_found;
             nonce_idx += blockDim.x)
        {
            if (((nonce_idx - base) & 0x1Fu) == 0 && work->found) {
                block_found = 1;
                break;
            }

            unsigned int nonce = static_cast<unsigned int>(nonce_idx);

            SHA256 hash;
            device_double_sha256_precomputed(&hash, nonce);

            if (device_little_endian_bit_comparison(hash.b, target, 32) < 0) {
                if (atomicCAS(&work->found, 0, 1) == 0) {
                    work->result = nonce;
                    __threadfence_system();
                }
                block_found = 1;
                break;
            }
        }

        __syncthreads();
        if (block_found) {
            return;
        }
    }
}

struct MiningContext {
    unsigned char *d_target{nullptr};
    DeviceWork *d_work{nullptr};
    DeviceWork h_work{};
    cudaStream_t stream{};
};

inline void init_mining_context(MiningContext &ctx, const unsigned char target[32])
{
    cudaMalloc(&ctx.d_target, 32);
    cudaMemcpy(ctx.d_target, target, 32, cudaMemcpyHostToDevice);
    cudaMalloc(&ctx.d_work, sizeof(DeviceWork));
    cudaStreamCreateWithFlags(&ctx.stream, cudaStreamNonBlocking);
}

inline void reset_mining_context(MiningContext &ctx, unsigned long long end_nonce)
{
    ctx.h_work.next_nonce = 0ULL;
    ctx.h_work.end_nonce = end_nonce;
    ctx.h_work.found = 0;
    ctx.h_work.result = 0xFFFFFFFFu;
}

inline void destroy_mining_context(MiningContext &ctx)
{
    cudaStreamDestroy(ctx.stream);
    cudaFree(ctx.d_target);
    cudaFree(ctx.d_work);
}

bool mine_block_with_context(HashBlock &block,
                             MiningContext &ctx,
                             unsigned long long base_chunk,
                             unsigned long long max_chunk,
                             int blocks_per_grid,
                             int threads_per_block)
{
    const unsigned long long total_nonces = 0x100000000ULL;
    reset_mining_context(ctx, total_nonces);

    WORD chunk0[16];
    WORD chunk1_base[16];
    prepare_sha256_chunks(block, chunk0, chunk1_base);
    cudaMemcpyToSymbol(d_chunk0_words, chunk0, sizeof(chunk0));
    cudaMemcpyToSymbol(d_chunk1_words_base, chunk1_base, sizeof(chunk1_base));

    unsigned int batch_size = static_cast<unsigned int>(base_chunk);
    if (batch_size == 0 || batch_size > 1 << 20) {
        batch_size = 4096;
    }

    cudaMemcpyAsync(ctx.d_work, &ctx.h_work, sizeof(DeviceWork),
                    cudaMemcpyHostToDevice, ctx.stream);

    mine_bitcoin_kernel<<<blocks_per_grid, threads_per_block, 0, ctx.stream>>>(
        ctx.d_target,
        ctx.d_work,
        batch_size
    );

    cudaMemcpyAsync(&ctx.h_work, ctx.d_work, sizeof(DeviceWork),
                    cudaMemcpyDeviceToHost, ctx.stream);

    cudaError_t err = cudaStreamSynchronize(ctx.stream);
    if (err != cudaSuccess) {
        block.nonce = 0;
        return false;
    }

    if (ctx.h_work.found && ctx.h_work.result != 0xFFFFFFFFu) {
        block.nonce = ctx.h_work.result;
        return true;
    }

    block.nonce = 0;
    return false;
}


void solve(FILE *fin, FILE *fout)
{
    char version[9];
    char prevhash[65];
    char ntime[9];
    char nbits[9];
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
    for(int i=0;i<tx;++i)
    {
        merkle_branch[i] = raw_merkle_branch + static_cast<size_t>(i) * 65;
        getline(merkle_branch[i], 65, fin);
        merkle_branch[i][64] = '\0';
    }

    unsigned char merkle_root[32];
    calc_merkle_root(merkle_root, tx, merkle_branch);

    HashBlock block;
    convert_string_to_little_endian_bytes((unsigned char *)&block.version, version, 8);
    convert_string_to_little_endian_bytes(block.prevhash,                  prevhash,    64);
    memcpy(block.merkle_root, merkle_root, 32);
    convert_string_to_little_endian_bytes((unsigned char *)&block.nbits,   nbits,     8);
    convert_string_to_little_endian_bytes((unsigned char *)&block.ntime,   ntime,     8);
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

    int threads_per_block = 256;
    int blocks_per_grid = 2048;

    MiningContext ctx;
    init_mining_context(ctx, target_hex);

    bool solved = mine_block_with_context(
        block,
        ctx,
        65536ULL,
        65536ULL,
        blocks_per_grid,
        threads_per_block);

    SHA256 sha256_ctx;
    double_sha256(&sha256_ctx, (unsigned char*)&block, 80);

    if(!(solved && little_endian_bit_comparison(sha256_ctx.b, target_hex, 32) < 0))
    {
        block.nonce = 0;
    }

    for(int i=0;i<4;++i)
    {
        fprintf(fout, "%02x", ((unsigned char*)&block.nonce)[i]);
    }
    fprintf(fout, "\n");

    destroy_mining_context(ctx);

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

    for (int i = 0; i < totalblock; ++i) {
        solve(fin, fout);
    }

    fclose(fin);
    fclose(fout);
    return 0;
}

