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
    // __host__ __device__ 表示這個函式可以被 host 和 device 端呼叫
    __host__ __device__ vec3() : x(0), y(0), z(0) {} //vec3()：預設把 x,y,z 設為 0
    //用傳進來的 xyz 初始化
    __host__ __device__ vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    //用傳進來的 v 初始化
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
//計算兩個 3D 向量的叉積（cross product），
//結果是一個同時垂直於兩個輸入向量的向量。常用來做座標系的 right/up/forward。
__host__ __device__ inline vec3 cross(const vec3& a, const vec3& b) {
    return vec3(a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x);
}
//一個向量函式，計算向量長度（歐幾里得距離）。
__host__ __device__ inline float length(const vec3& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}
//__host__ __device__ :CPU/GPU 都可以呼叫的函式
//inline : 提示編譯器把這個函式展開成 inline 函式，避免函式呼叫的 overhead
//一般函式的呼叫流程:
//1.保存當前執行狀態（如：register）。
//2.將參數 push to stack 中。
//3.跳轉到函數的代碼位置。
//4.執行函數。
//5.回傳結果並恢復呼叫點的狀態。若為小型函數，呼叫的開銷可能比函數本身的執行時間大。
//使用inline 
//編譯器會把函數的程式碼副本放置在每個呼叫該函數的地方。
// 沒有function call、stack push/pop 的 overhead，直接執行函數的程式碼。
//在GPU中，因為是SIMT，所以當使用inline的時候，可以避免額外的 call / return 指令與控制流程跳轉
//const vec3& v : 用 reference 傳遞，避免複製整個 vec3
//const：保證在 normalize() 裡不會改動呼叫者傳進來的那個 vec3
//float len = length(v) : 計算向量長度
//if (len > 0.0f) : 如果向量長度大於 0，則正規化
//return vec3(v.x / len, v.y / len, v.z / len) : 正規化後的向量
//return vec3(0.0f, 0.0f, 0.0f) : 如果向量長度為 0，則返回 0 向量
//一個向量函式，把輸入向量變成「單位向量」（長度變成 1、方向不變）。
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
//GPU 的 常數記憶體（constant memory），在 device（GPU）端，唯讀、快取、所有 threads 共用
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
//希望編譯器假設 每個 block 最多 256 threads
//希望 每個 SM（Streaming Multiprocessor）至少能同時駐留 4 個 blocks。
//目的:用較少暫存器 → 讓 SM 同時跑更多 blocks → 提高吞吐
__launch_bounds__(256, 4)
//由 CPU 端用 <<<grid, block>>> 啟動
//__global__:GPU kernel 的宣告，在 device 上跑，每個 thread 會跑這個函式一次
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
//在 GPU 算完、已經在 CPU 記憶體裡的 RGBA 影像，寫成一個 PNG 檔案
void write_png(const char* filename, unsigned char* image, unsigned int width, unsigned int height) {
    unsigned error = lodepng_encode32_file(filename, image, width, height);
    if (error) printf("png error %u: %s\n", error, lodepng_error_text(error));
}

int main(int argc, char** argv) {
    //作業規格要求執行方式是： ./hw3 x1 y1 z1 x2 y2 z2 width height filename
    //這是 9 個參數 + 程式名，共 10 個字串，所以用 assert 直接檢查；不符就直接中止（避免後面讀 argv 越界）。
    //argc = argument count, 命令列參數的「數量」，argc == 10 表示有 9 個參數 + 程式名
    //argv = argument vector 
    assert(argc == 10);

    //camera_pos_*：相機位置 (x1,y1,z1)
    float camera_pos_x = atof(argv[1]);
    float camera_pos_y = atof(argv[2]);
    float camera_pos_z = atof(argv[3]);
    //target_pos_*：相機看的目標點 (x2,y2,z2)
    float target_pos_x = atof(argv[4]);
    float target_pos_y = atof(argv[5]);
    float target_pos_z = atof(argv[6]);
    //width：輸出影像大小
    unsigned int width = atoi(argv[7]);
    unsigned int height = atoi(argv[8]);
    float iResolution_x = width;
    float iResolution_y = height;
    //把 host 端的值copy到 device 端的 __constant__ 變數（例如 d_camera_pos_x）。
    //cudaMemcpyToSymbol(symbol, src, size);
    //symbol:device 端的 __constant__ 變數名稱
    //src:host 端資料的位址
    //size:複製的位元組數
    cudaMemcpyToSymbol(d_camera_pos_x, &camera_pos_x, sizeof(float));
    cudaMemcpyToSymbol(d_camera_pos_y, &camera_pos_y, sizeof(float));
    cudaMemcpyToSymbol(d_camera_pos_z, &camera_pos_z, sizeof(float));
    cudaMemcpyToSymbol(d_target_pos_x, &target_pos_x, sizeof(float));
    cudaMemcpyToSymbol(d_target_pos_y, &target_pos_y, sizeof(float));
    cudaMemcpyToSymbol(d_target_pos_z, &target_pos_z, sizeof(float));
    cudaMemcpyToSymbol(d_iResolution_x, &iResolution_x, sizeof(float));
    cudaMemcpyToSymbol(d_iResolution_y, &iResolution_y, sizeof(float));

    // 在 CPU 端先把相機座標系（camera basis）算好，避免每個 pixel 重算
    //型別是 vec3 變數 h_camera_pos constructor 初始化，建立物件
    vec3 h_camera_pos(camera_pos_x, camera_pos_y, camera_pos_z);
    vec3 h_target_pos(target_pos_x, target_pos_y, target_pos_z);
    vec3 h_cf = normalize(h_target_pos - h_camera_pos);
    if (length(h_cf) < 1e-6f) {
        h_cf = vec3(0.0f, 0.0f, -1.0f);
    }
    //up_vector:變數名稱 會呼叫 struct vec3的 constructor 初始化，建立物件
    vec3 up_vector(0.0f, 1.0f, 0.0f);
    // inline function
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
    //make_float3 是 CUDA 提供的 helper function（在你 #include <cuda_runtime.h> 後可用）
    //把三個 float 打包成 CUDA 的內建型別 float3（通常用來跟 CUDA API/內建向量型別搭配）
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
    //cudaMalloc(&ptr, size) 是 CUDA 的「配置 GPU 記憶體」API。
    //在 GPU (device) 上分配一塊連續記憶體，大小是 size bytes，並把那塊記憶體的位址寫到 ptr。
    cudaMalloc(&d_image, image_size);

    //決定 GPU 上的「執行配置」：每個 block 幾個 thread（blockDim），以及總共需要幾個 block（gridDim），讓 thread 數量足夠覆蓋整張影像的所有像素。
    //CUDA kernel 是「大量 thread 並行」在跑；你要告訴 CUDA：「每個 block 放幾個 thread」以及「總共要多少個 block」，才能讓 GPU 產生足夠的 threads 來處理 (width * height) 個像素。
    //dim3 是 CUDA 的 3D 維度型別（有 x,y,z 欄位）。這裡等同於 blockDim.x=16, blockDim.y=16, blockDim.z=1
    //blockDim(16,16):這是常見的 2D 影像配置：一個 block 處理一塊 16×16 的像素區塊（256 threads）。
    dim3 blockDim(16, 16);
    //向上取整，即 ceil(a/b)width 不一定是 16 的倍數
    //(width + blockDim.x - 1) / blockDim.x 會把「除不盡的那一點」也算進去，確保最右邊/最下面的像素也有 thread 覆蓋到。
    //多出來的 threads 會在 kernel 裡用 if (i>=height || j>=width) return; 早退
    //「block 的數量」（多少個 block 沿 x/y 排列）
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);
    //啟動 CUDA kernel：呼叫 render_kernel 讓 GPU 真正開始算每個像素的顏色。
    render_kernel<<<gridDim, blockDim>>>(d_image, width, height);
    //立刻檢查 kernel launch 有沒有失敗：如果啟動就出錯（例如資源不足、參數不對），馬上印錯誤並退出。
    //kernel launch 通常是非同步的，但「啟動當下」就可能失敗（例如 block 太大、shared memory/regs 超限、傳參不合法）。
    //這行能立即抓到 launch 錯誤，否則你可能到後面才發現結果全錯或程式怪怪的。
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    //cudaDeviceSynchronize() 是 CUDA 的「等待 GPU 完成工作」API。
    cudaDeviceSynchronize();
    //在 CPU（host）記憶體配置一個陣列，用來接住 GPU 計算好的影像。image_size = width*height*4（RGBA）。
    unsigned char* h_image = new unsigned char[image_size];
    //把 GPU 計算好的影像，複製回 CPU 端的 h_image 陣列。
    cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost);
    //把 h_image 這個 RGBA buffer 編碼成 PNG，寫到你命令列第 9 個參數指定的檔名
    write_png(argv[9], h_image, width, height);
    //釋放 GPU 記憶體。
    cudaFree(d_image);
    delete[] h_image;

    return 0;
}
