#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <lodepng.h>
#include <cuda_runtime.h>

#define pi 3.1415926535897932384626433832795

// Custom vector types and operations for CUDA
struct vec2 {
    float x, y;
    __host__ __device__ vec2() : x(0), y(0) {}
    __host__ __device__ vec2(float x, float y) : x(x), y(y) {}
    __host__ __device__ vec2(float v) : x(v), y(v) {}
};

struct vec3 {
    float x, y, z;
    __host__ __device__ vec3() : x(0), y(0), z(0) {}
    __host__ __device__ vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    __host__ __device__ vec3(float v) : x(v), y(v), z(v) {}
};

// Vector operations
__host__ __device__ inline vec2 operator+(const vec2& a, const vec2& b) {
    return vec2(a.x + b.x, a.y + b.y);
}

__host__ __device__ inline vec2 operator*(const vec2& a, float s) {
    return vec2(a.x * s, a.y * s);
}

__host__ __device__ inline vec2 operator*(float s, const vec2& a) {
    return vec2(a.x * s, a.y * s);
}

__host__ __device__ inline vec2 operator/(const vec2& a, float s) {
    return vec2(a.x / s, a.y / s);
}

__host__ __device__ inline vec3 operator+(const vec3& a, const vec3& b) {
    return vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline vec3 operator-(const vec3& a, const vec3& b) {
    return vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline vec3 operator*(const vec3& a, float s) {
    return vec3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline vec3 operator*(float s, const vec3& a) {
    return vec3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline vec3 operator*(const vec3& a, const vec3& b) {
    return vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ inline vec3 operator/(const vec3& a, float s) {
    return vec3(a.x / s, a.y / s, a.z / s);
}

__host__ __device__ inline void operator+=(vec3& a, const vec3& b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
}

// Vector functions
__host__ __device__ inline float dot(const vec3& a, const vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline vec3 cross(const vec3& a, const vec3& b) {
    return vec3(a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x);
}

__host__ __device__ inline float length(const vec3& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__host__ __device__ inline vec3 normalize(const vec3& v) {
    float len = length(v);
    if (len > 0.0f) {
        return vec3(v.x / len, v.y / len, v.z / len);
    }
    return vec3(0.0f, 0.0f, 0.0f);
}

__host__ __device__ inline float vmin(float a, float b) {
    return a < b ? a : b;
}

__host__ __device__ inline float vmax(float a, float b) {
    return a > b ? a : b;
}

__host__ __device__ inline float clamp(float x, float minVal, float maxVal) {
    return vmin(vmax(x, minVal), maxVal);
}

__host__ __device__ inline vec3 vmin(const vec3& a, float b) {
    return vec3(vmin(a.x, b), vmin(a.y, b), vmin(a.z, b));
}

__host__ __device__ inline vec3 clamp(const vec3& v, float minVal, float maxVal) {
    return vec3(clamp(v.x, minVal, maxVal),
                clamp(v.y, minVal, maxVal),
                clamp(v.z, minVal, maxVal));
}

__host__ __device__ inline vec3 vcos(const vec3& v) {
    return vec3(cosf(v.x), cosf(v.y), cosf(v.z));
}

__host__ __device__ inline vec3 vpow(const vec3& v, const vec3& p) {
    return vec3(powf(v.x, p.x), powf(v.y, p.y), powf(v.z, p.z));
}

// GLM-like swizzle functions
__host__ __device__ inline vec3 xyy(const vec2& v) { return vec3(v.x, v.y, v.y); }
__host__ __device__ inline vec3 yxy(const vec2& v) { return vec3(v.y, v.x, v.y); }
__host__ __device__ inline vec3 yyx(const vec2& v) { return vec3(v.y, v.y, v.x); }

// Constants
constexpr int AA = 3;
constexpr int MD_ITER = 24;
constexpr int RAY_STEP = 10000;
constexpr int SHADOW_STEP = 1500;
__constant__ float power = 8.0f;
__constant__ float step_limiter = 0.2f;
__constant__ float ray_multiplier = 0.1f;
__constant__ float bailout = 2.0f;
__constant__ float eps = 0.0005f;
__constant__ float FOV = 1.5f;
__constant__ float far_plane = 100.0f;

// Store camera and resolution in constant memory
__constant__ float d_camera_pos_x, d_camera_pos_y, d_camera_pos_z;
__constant__ float d_target_pos_x, d_target_pos_y, d_target_pos_z;
__constant__ float d_iResolution_x, d_iResolution_y;
__constant__ float3 d_cf;
__constant__ float3 d_cs;
__constant__ float3 d_cu;
__constant__ float3 d_sd;

// Mandelbulb distance function
__device__ float md(const vec3& p, float& trap) {
    vec3 v = p;
    float dr = 1.0;
    float r = length(v);
    trap = r;

    for (int i = 0; i < MD_ITER; ++i) {
        float inv_r = fmaxf(r, 1e-6f);
        inv_r = 1.0f / inv_r;

        float theta = atan2f(v.y, v.x) * power;
        float phi = asinf(fmaxf(fminf(v.z * inv_r, 1.0f), -1.0f)) * power;

        float sinTheta, cosTheta;
        float sinPhi, cosPhi;
        sincosf(theta, &sinTheta, &cosTheta);
        sincosf(phi, &sinPhi, &cosPhi);

        float r2 = r * r;
        float r4 = r2 * r2;
        float r6 = r4 * r2;
        float r7 = r6 * r;
        float r8 = r4 * r4;

        dr = power * r7 * dr + 1.0f;
        v = p + r8 * vec3(cosTheta * cosPhi,
                          cosPhi * sinTheta,
                          -sinPhi);

        trap = vmin(trap, r);

        r = length(v);
        if (r > bailout) break;
    }
    return 0.5 * logf(r) * r / dr;
}

// Scene mapping - 90 degree rotation around X-axis: (x,y,z) -> (x,-z,y)
__device__ float map(const vec3& p, float& trap, int& ID) {
    vec3 rp = vec3(p.x, -p.z, p.y);  // 90 deg rotation, no trig functions needed!
    ID = 1;
    return md(rp, trap);
}

__device__ float map(const vec3& p) {
    float dmy;
    int dmy2;
    return map(p, dmy, dmy2);
}

// Palette function
__device__ vec3 pal(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b * vcos(2.0 * pi * (c * t + d));
}

// Soft shadow
__device__ float softshadow(const vec3& ro, const vec3& rd, float k) {
    float res = 1.0f;
    float t = 0.0f;
    for (int i = 0; i < SHADOW_STEP; ++i) {
        float h = map(ro + rd * t);
        if (t > 0.0f) {
            float candidate = k * h / t;
            res = fminf(res, candidate);
            if (res <= 0.02f) return 0.02f;
            if (res >= 0.99f && h >= 0.5f && t >= 5.0f) return 1.0f;
        }
        t += clamp(h, 0.001f, step_limiter);
        if (t > far_plane) break;
    }
    return clamp(res, 0.02f, 1.0f);
}

// Calculate surface normal
__device__ vec3 calcNor(const vec3& p) {
    vec2 e = vec2(eps, 0.0f);
    return normalize(vec3(
        map(p + xyy(e)) - map(p - xyy(e)),
        map(p + yxy(e)) - map(p - yxy(e)),
        map(p + yyx(e)) - map(p - yyx(e))
    ));
}

// Ray tracing
__device__ float trace(const vec3& ro, const vec3& rd, float& trap, int& ID) {
    float t = 0;
    float len = 0;

    for (int i = 0; i < RAY_STEP; ++i) {
        len = map(ro + rd * t, trap, ID);
        if (fabsf(len) < eps || t > far_plane) break;
        t += len * ray_multiplier;
    }
    return t < far_plane ? t : -1.0;
}

// Main kernel
__launch_bounds__(256, 4)
__global__ void render_kernel(unsigned char* image, unsigned int width, unsigned int height) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= height || j >= width) return;

    // Pre-compute values that don't change across AA samples
    const vec3 camera_pos = vec3(d_camera_pos_x, d_camera_pos_y, d_camera_pos_z);
    const vec3 cf = vec3(d_cf.x, d_cf.y, d_cf.z);
    const vec3 cs = vec3(d_cs.x, d_cs.y, d_cs.z);
    const vec3 cu = vec3(d_cu.x, d_cu.y, d_cu.z);
    const vec3 sd = vec3(d_sd.x, d_sd.y, d_sd.z);
    const vec3 sc = vec3(1., .9, .717);
    const vec3 ambc = vec3(0.3);
    const float gloss = 32.0;

    vec3 fcol(0.0);

    const float invAA = 1.0f / (float)AA;
    const vec2 resolution = vec2(d_iResolution_x, d_iResolution_y);
    const vec2 screen_origin = vec2(-d_iResolution_x, -d_iResolution_y);
    const vec2 pixel_base = vec2((float)j, (float)i);

    for (int m = 0; m < AA; ++m) {
        for (int n = 0; n < AA; ++n) {
            vec2 p = vec2((float)m, (float)n) * invAA;
            vec2 uv = (screen_origin + 2.0f * (pixel_base + p)) / resolution.y;
            uv.y *= -1;

            vec3 rd = normalize(uv.x * cs + uv.y * cu + FOV * cf);

            float trap;
            int objID;
            float d = trace(camera_pos, rd, trap, objID);

            vec3 col(0.0);
            if (d >= 0.) {
                vec3 pos = camera_pos + rd * d;
                vec3 nr = calcNor(pos);
                vec3 hal = normalize(sd - rd);

                col = pal(trap - 0.4, vec3(0.5), vec3(0.5), vec3(1.0), vec3(0.0, 0.1, 0.2));
                
                float amb = (0.7 + 0.3 * nr.y) * (0.2 + 0.8 * clamp(0.05 * logf(trap), 0.0, 1.0));
                float sdw = softshadow(pos + 0.001 * nr, sd, 16.0);
                float dif = clamp(dot(sd, nr), 0.0, 1.0) * sdw;
                float spe = powf(clamp(dot(nr, hal), 0.0, 1.0), gloss) * dif;

                vec3 lin(0.0);
                lin += ambc * (0.05 + 0.95 * amb);
                lin += sc * dif * 0.8;
                col = col * lin;

                col = vpow(col, vec3(0.7, 0.9, 1.0));
                col += vec3(spe * 0.8);
            }

            fcol += clamp(vpow(col, vec3(0.4545)), 0.0, 1.0);
        }
    }

    fcol = fcol * (255.0 / (float)(AA * AA));

    int idx = i * width * 4 + j * 4;
    image[idx + 0] = (unsigned char)fcol.x;
    image[idx + 1] = (unsigned char)fcol.y;
    image[idx + 2] = (unsigned char)fcol.z;
    image[idx + 3] = 255;
}

void write_png(const char* filename, unsigned char* image, unsigned int width, unsigned int height) {
    unsigned error = lodepng_encode32_file(filename, image, width, height);
    if (error) printf("png error %u: %s\n", error, lodepng_error_text(error));
}

int main(int argc, char** argv) {
    //作業規格要求執行方式是： ./hw3 x1 y1 z1 x2 y2 z2 width height filename
    //這是 9 個參數 + 程式名，共 10 個字串，所以用 assert 直接檢查；不符就直接中止（避免後面讀 argv 越界）。
    assert(argc == 10);

    float camera_pos_x = atof(argv[1]);
    float camera_pos_y = atof(argv[2]);
    float camera_pos_z = atof(argv[3]);
    float target_pos_x = atof(argv[4]);
    float target_pos_y = atof(argv[5]);
    float target_pos_z = atof(argv[6]);
    unsigned int width = atoi(argv[7]);
    unsigned int height = atoi(argv[8]);
    float iResolution_x = width;
    float iResolution_y = height;

    cudaMemcpyToSymbol(d_camera_pos_x, &camera_pos_x, sizeof(float));
    cudaMemcpyToSymbol(d_camera_pos_y, &camera_pos_y, sizeof(float));
    cudaMemcpyToSymbol(d_camera_pos_z, &camera_pos_z, sizeof(float));
    cudaMemcpyToSymbol(d_target_pos_x, &target_pos_x, sizeof(float));
    cudaMemcpyToSymbol(d_target_pos_y, &target_pos_y, sizeof(float));
    cudaMemcpyToSymbol(d_target_pos_z, &target_pos_z, sizeof(float));
    cudaMemcpyToSymbol(d_iResolution_x, &iResolution_x, sizeof(float));
    cudaMemcpyToSymbol(d_iResolution_y, &iResolution_y, sizeof(float));

    vec3 h_camera_pos(camera_pos_x, camera_pos_y, camera_pos_z);
    vec3 h_target_pos(target_pos_x, target_pos_y, target_pos_z);
    vec3 h_cf = normalize(h_target_pos - h_camera_pos);
    if (length(h_cf) < 1e-6f) {
        h_cf = vec3(0.0f, 0.0f, -1.0f);
    }

    vec3 up_vector(0.0f, 1.0f, 0.0f);
    vec3 side_candidate = cross(h_cf, up_vector);
    if (length(side_candidate) < 1e-4f) {
        up_vector = vec3(0.0f, 0.0f, 1.0f);
        side_candidate = cross(h_cf, up_vector);
    }

    vec3 h_cs = normalize(side_candidate);
    vec3 h_cu = normalize(cross(h_cs, h_cf));
    if (length(h_cu) < 1e-6f) {
        h_cu = vec3(0.0f, 1.0f, 0.0f);
    }

    float cam_length = length(h_camera_pos);
    vec3 h_sd = cam_length > 1e-6f ? h_camera_pos / cam_length : vec3(0.0f, 1.0f, 0.0f);

    float3 cf_const = make_float3(h_cf.x, h_cf.y, h_cf.z);
    float3 cs_const = make_float3(h_cs.x, h_cs.y, h_cs.z);
    float3 cu_const = make_float3(h_cu.x, h_cu.y, h_cu.z);
    float3 sd_const = make_float3(h_sd.x, h_sd.y, h_sd.z);

    cudaMemcpyToSymbol(d_cf, &cf_const, sizeof(float3));
    cudaMemcpyToSymbol(d_cs, &cs_const, sizeof(float3));
    cudaMemcpyToSymbol(d_cu, &cu_const, sizeof(float3));
    cudaMemcpyToSymbol(d_sd, &sd_const, sizeof(float3));

    unsigned char* d_image;
    size_t image_size = width * height * 4 * sizeof(unsigned char);
    cudaMalloc(&d_image, image_size);

    // Use 16x16 blocks to reduce per-block register pressure and improve SM occupancy
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    render_kernel<<<gridDim, blockDim>>>(d_image, width, height);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceSynchronize();

    unsigned char* h_image = new unsigned char[image_size];
    cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost);

    write_png(argv[9], h_image, width, height);

    cudaFree(d_image);
    delete[] h_image;

    return 0;
}
