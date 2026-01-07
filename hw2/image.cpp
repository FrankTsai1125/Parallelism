#include <cmath>
#include <iostream>
#include <cassert>
#include <utility>
#include <omp.h>

#include "image.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

Image::Image(std::string file_path)
{
    unsigned char *img_data = stbi_load(file_path.c_str(), &width, &height, &channels, 0);
    if (img_data == nullptr) {
        const char *error_msg = stbi_failure_reason();
        std::cerr << "Failed to load image: " << file_path.c_str() << "\n";
        std::cerr << "Error msg (stb_image): " << error_msg << "\n";
        std::exit(1);
    }

    size = width * height * channels;
    data = new float[size]; 
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int c = 0; c < channels; c++) {
                int src_idx = y*width*channels + x*channels + c;
                int dst_idx = c*height*width + y*width + x;
                data[dst_idx] = img_data[src_idx] / 255.;
            }
        }
    }
    if (channels == 4)
        channels = 3; //ignore alpha channel
    stbi_image_free(img_data);
}

Image::Image(int w, int h, int c)
    :width {w},
     height {h},
     channels {c},
     size {w*h*c},
     data {new float[w*h*c]()}
{
}

Image::Image()
    :width {0},
     height {0},
     channels {0},
     size {0},
     data {nullptr} 
{
}

Image::~Image()
{
    delete[] this->data;
}

Image::Image(const Image& other)
    :width {other.width},
     height {other.height},
     channels {other.channels},
     size {other.size},
     data {new float[other.size]}
{
    //std::cout << "copy constructor\n";
    for (int i = 0; i < size; i++)
        data[i] = other.data[i];
}

Image& Image::operator=(const Image& other)
{
    if (this != &other) {
        delete[] data;
        //std::cout << "copy assignment\n";
        width = other.width;
        height = other.height;
        channels = other.channels;
        size = other.size;
        data = new float[other.size];
        for (int i = 0; i < other.size; i++)
            data[i] = other.data[i];
    }
    return *this;
}

Image::Image(Image&& other)
    :width {other.width},
     height {other.height},
     channels {other.channels},
     size {other.size},
     data {other.data}
{
    //std::cout << "move constructor\n";
    other.data = nullptr;
    other.size = 0;
}

Image& Image::operator=(Image&& other)
{
    //std::cout << "move assignment\n";
    delete[] data;
    data = other.data;
    width = other.width;
    height = other.height;
    channels = other.channels;
    size = other.size;

    other.data = nullptr;
    other.size = 0;
    return *this;
}

//save image as jpg file
bool Image::save(std::string file_path)
{
    unsigned char *out_data = new unsigned char[width*height*channels]; 
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int c = 0; c < channels; c++) {
                int dst_idx = y*width*channels + x*channels + c;
                int src_idx = c*height*width + y*width + x;
                out_data[dst_idx] = std::roundf(data[src_idx] * 255.);
            }
        }
    }
    bool success = stbi_write_jpg(file_path.c_str(), width, height, channels, out_data, 100);
    if (!success)
        std::cerr << "Failed to save image: " << file_path << "\n";

    delete[] out_data;
    return true;
}

void Image::set_pixel(int x, int y, int c, float val)
{
    if (x >= width || x < 0 || y >= height || y < 0 || c >= channels || c < 0) {
        std::cerr << "set_pixel() error: Index out of bounds.\n";
        std::exit(1);
    }
    data[c*width*height + y*width + x] = val;
}

float Image::get_pixel(int x, int y, int c) const
{
    if (x < 0)
        x = 0;
    if (x >= width)
        x = width - 1;
    if (y < 0)
        y = 0;
    if (y >= height)
        y = height - 1;
    return data[c*width*height + y*width + x];
}

void Image::clamp()
{
    int size = width * height * channels;
    for (int i = 0; i < size; i++) {
        float val = data[i];
        val = (val > 1.0) ? 1.0 : val;
        val = (val < 0.0) ? 0.0 : val;
        data[i] = val;
    }
}

//map coordinate from 0-current_max range to 0-new_max range
float map_coordinate(float new_max, float current_max, float coord)
{
    float a = new_max / current_max;
    float b = -0.5 + a*0.5;
    return a*coord + b;
}

Image Image::resize(int new_w, int new_h, Interpolation method) const
{
    Image resized(new_w, new_h, this->channels);
    
    if (method == Interpolation::NEAREST) {
        // Optimized nearest neighbor
        //parallel for：建立一個 thread team，並把後面的迴圈工作分配給 threads。
        // collapse(2)：把兩層巢狀迴圈 for y + for x 「攤平」成一個 2D iteration space 交給 OpenMP 分配。
        // 等價概念：把每個 (y,x) 當成一個獨立工作單元，總工作數約是 new_h * new_w。
        // 好處：若 new_h 不大或某些 y 行工作不均，collapse 能讓分配更平均。
        // schedule(static)：靜態排程。OpenMP 會在迴圈開始前就把「固定範圍」的 (y,x) 工作切給每個 thread。
        // 好處：排程 overhead 最低、cache locality 常常也比較好。
        // 代價：若不同 (y,x) 工作量差異很大，static 可能不平衡；但 resize 每個 pixel 工作量幾乎一樣，所以 static 很適合。
        #pragma omp parallel for collapse(2) schedule(static)
        for (int y = 0; y < new_h; y++) {
            for (int x = 0; x < new_w; x++) {
                float old_x = map_coordinate(this->width, new_w, x);
                float old_y = map_coordinate(this->height, new_h, y);
                int src_x = std::round(old_x);
                int src_y = std::round(old_y);
                // Clamp
                if (src_x < 0) src_x = 0;
                if (src_x >= this->width) src_x = this->width - 1;
                if (src_y < 0) src_y = 0;
                if (src_y >= this->height) src_y = this->height - 1;
                
                for (int c = 0; c < this->channels; c++) {
                    resized.data[c * new_h * new_w + y * new_w + x] = 
                        this->data[c * this->height * this->width + src_y * this->width + src_x];
                }
            }
        }
    } else {
        // Bilinear interpolation
        #pragma omp parallel for collapse(2) schedule(static)
        for (int y = 0; y < new_h; y++) {
            for (int x = 0; x < new_w; x++) {
                float old_x = map_coordinate(this->width, new_w, x);
                float old_y = map_coordinate(this->height, new_h, y);
                
                for (int c = 0; c < this->channels; c++) {
                    float value = bilinear_interpolate(*this, old_x, old_y, c);
                    resized.data[c * new_h * new_w + y * new_w + x] = value;
                }
            }
        }
    }
    return resized;
}

float bilinear_interpolate(const Image& img, float x, float y, int c)
{
    float p1, p2, p3, p4, q1, q2;
    float x_floor = std::floor(x), y_floor = std::floor(y);
    float x_ceil = x_floor + 1, y_ceil = y_floor + 1;
    p1 = img.get_pixel(x_floor, y_floor, c);
    p2 = img.get_pixel(x_ceil, y_floor, c);
    p3 = img.get_pixel(x_floor, y_ceil, c);
    p4 = img.get_pixel(x_ceil, y_ceil, c);
    q1 = (y_ceil-y)*p1 + (y-y_floor)*p3;
    q2 = (y_ceil-y)*p2 + (y-y_floor)*p4;
    return (x_ceil-x)*q1 + (x-x_floor)*q2;
}

float nn_interpolate(const Image& img, float x, float y, int c)
{
    return img.get_pixel(std::round(x), std::round(y), c);
}

Image rgb_to_grayscale(const Image& img)
{
    assert(img.channels == 3);
    Image gray(img.width, img.height, 1);
    
    int w = img.width;
    int h = img.height;
    const float* r_data = img.data;
    const float* g_data = img.data + w * h;
    const float* b_data = img.data + 2 * w * h;
    float* gray_data = gray.data;
    //平行化，使用static method，因為每個 pixel 的工作量幾乎一樣。
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < w * h; idx++) {
        gray_data[idx] = 0.299f * r_data[idx] + 
                        0.587f * g_data[idx] + 
                        0.114f * b_data[idx];
    }
    return gray;
}

Image grayscale_to_rgb(const Image& img)
{
    assert(img.channels == 1);
    Image rgb(img.width, img.height, 3);
    for (int x = 0; x < img.width; x++) {
        for (int y = 0; y < img.height; y++) {
            float gray_val = img.get_pixel(x, y, 0);
            rgb.set_pixel(x, y, 0, gray_val);
            rgb.set_pixel(x, y, 1, gray_val);
            rgb.set_pixel(x, y, 2, gray_val);
        }
    }
    return rgb;
}

// separable 2D gaussian blur for 1 channel image
Image gaussian_blur(const Image& img, float sigma)
{
    return gaussian_blur(img, sigma, nullptr);
}

// Memory-optimized version: reuses tmp buffer to avoid repeated allocations
//新增外部暫存 buffer 指標（供呼叫端重複使用），省配置/釋放成本（效能提升）
Image gaussian_blur(const Image& img, float sigma, Image* reuse_tmp)
{
    assert(img.channels == 1); //只支援單通道（灰階）圖片

    int size = std::ceil(6 * sigma);
    if (size % 2 == 0)
        size++;
    int center = size / 2;
    Image kernel(size, 1, 1);
    float sum = 0;
    for (int k = -size/2; k <= size/2; k++) {
        float val = std::exp(-(k*k) / (2*sigma*sigma));
        kernel.set_pixel(center+k, 0, 0, val);
        sum += val;
    }
    for (int k = 0; k < size; k++)
        kernel.data[k] /= sum;

    // Reuse tmp buffer if provided and size matches
    //重用 tmp 可以降低大量配置成本（pyramid 內 blur 會跑很多次）。
    Image tmp;
    if (reuse_tmp && reuse_tmp->width == img.width && 
        reuse_tmp->height == img.height && reuse_tmp->channels == 1) {
        tmp = std::move(*reuse_tmp);  // Take ownership
    } else {
        tmp = Image(img.width, img.height, 1);  // Allocate new
    }
    
    Image filtered(img.width, img.height, 1);

    int w = img.width;
    int h = img.height;
    const float* img_data = img.data;
    float* tmp_data = tmp.data;
    float* filt_data = filtered.data;
    const float* kern_data = kernel.data;

    // convolve vertical - optimized for cache and vectorization
    // Handle border separately for better performance
    //分開處理邊界可以避免在迴圈內做邊界檢查，提高效能。
    //開始平行化
    //parallel 區塊內的 code 會被平行化，但是要注意會有 race condition，所以需要用 reduction。
    //啟動多個 thread 來處理。
    //OpenMP 會 建立一個 thread team（預設 thread 數量由：OMP_NUM_THREADS、或omp_set_num_threads()、或runtime/系統預設）
    //這些 threads 會一起執行大括號 { ... } 裡的程式碼。
    //但不是每個 thread 永遠固定只做 Interior/Top/Bottom 其中之一。
    #pragma omp parallel
    {
        // Interior (no boundary checks needed)
        //把 y=center..h-center-1 的 rows 靜態分配給 thread，
        //算完不需要 barrier（因為接下來 top/bottom 
        //也在同一 parallel 區塊內，各自獨立寫不同 y）。
        //static 表示分配好就不會再動，
        //nowait 表示不需要等待所有 thread 都完成。
        //在 parallel 區塊裡，每遇到一個 #pragma omp for：
        //它不是再開一組新的 threads。
        //它是把「接下來那個 for 迴圈的 iteration（這裡是 y 的範圍）」分配給同一個 thread 去做。
        //每一個thread 都有自己個center 值
        //每一個thread 負責的 y 範圍是 center..h-center-1
        //有些 threads 很快做完 Interior 自己的 y-chunk → 立刻去做 Top border 的 y-chunk
        //同時間可能還有其他 threads 還在跑 Interior 的 y-chunk。
        #pragma omp for schedule(static) nowait
        for (int y = center; y < h - center; y++) {
            int y_base = y * w;
            for (int x = 0; x < w; x++) {
                float sum = 0.0f;
                //允許編譯器向量化 k-loop，並做 reduction
                //這是 SIMD 向量化的 reduction（同一個 thread 裡，向量 lanes 的加總），
                //不是 OpenMP threads 之間的 atomic add。
                #pragma omp simd reduction(+:sum)
                for (int k = 0; k < size; k++) {
                    sum += img_data[(y - center + k) * w + x] * kern_data[k];
                }
                tmp_data[y_base + x] = sum;
            }
        }
        
        // Top border
        //把 y=0..center-1 的 rows 靜態分配給 threads，
        //算完不需要 barrier（因為接下來 bottom 
        //也在同一 parallel 區塊內，各自獨立寫不同 y）。
        #pragma omp for schedule(static) nowait
        for (int y = 0; y < center; y++) {
            int y_base = y * w;
            for (int x = 0; x < w; x++) {
                float sum = 0.0f;
                for (int k = 0; k < size; k++) {
                    int sy = y - center + k;
                    sy = (sy < 0) ? 0 : sy;
                    sum += img_data[sy * w + x] * kern_data[k];
                }
                tmp_data[y_base + x] = sum;
            }
        }
        
        // Bottom border
        //把 y=h-center..h-1 的 rows 靜態分配給 threads，
        //算完不需要 barrier（因為接下來 top 
        //也在同一 parallel 區塊內，各自獨立寫不同 y）。
        #pragma omp for schedule(static) nowait
        for (int y = h - center; y < h; y++) {
            int y_base = y * w;
            for (int x = 0; x < w; x++) {
                float sum = 0.0f;
                for (int k = 0; k < size; k++) {
                    int sy = y - center + k;
                    sy = (sy >= h) ? h - 1 : sy;
                    sum += img_data[sy * w + x] * kern_data[k];
                }
                tmp_data[y_base + x] = sum;
            }
        }
    }
    
    // convolve horizontal - optimized for cache and vectorization
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < h; y++) {
        int y_base = y * w;
        const float* tmp_row = tmp_data + y_base;
        float* filt_row = filt_data + y_base;
        
        // Interior (no boundary checks)
        for (int x = center; x < w - center; x++) {
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < size; k++) {
                sum += tmp_row[x - center + k] * kern_data[k];
            }
            filt_row[x] = sum;
        }
        
        // Left border
        for (int x = 0; x < center; x++) {
            float sum = 0.0f;
            for (int k = 0; k < size; k++) {
                int sx = x - center + k;
                sx = (sx < 0) ? 0 : sx;
                sum += tmp_row[sx] * kern_data[k];
            }
            filt_row[x] = sum;
        }
        
        // Right border
        for (int x = w - center; x < w; x++) {
            float sum = 0.0f;
            for (int k = 0; k < size; k++) {
                int sx = x - center + k;
                sx = (sx >= w) ? w - 1 : sx;
                sum += tmp_row[sx] * kern_data[k];
            }
            filt_row[x] = sum;
        }
    }
    
    // Return tmp buffer to caller for reuse
    if (reuse_tmp) {
        *reuse_tmp = std::move(tmp);
    }
    
    return filtered;
}

void draw_point(Image& img, int x, int y, int size)
{
    for (int i = x-size/2; i <= x+size/2; i++) {
        for (int j = y-size/2; j <= y+size/2; j++) {
            if (i < 0 || i >= img.width) continue;
            if (j < 0 || j >= img.height) continue;
            if (std::abs(i-x) + std::abs(j-y) > size/2) continue;
            if (img.channels == 3) {
                img.set_pixel(i, j, 0, 1.f);
                img.set_pixel(i, j, 1, 0.f);
                img.set_pixel(i, j, 2, 0.f);
            } else {
                img.set_pixel(i, j, 0, 1.f);
            }
        }
    }
}

void draw_line(Image& img, int x1, int y1, int x2, int y2)
{
    if (x2 < x1) {
        std::swap(x1, x2);
        std::swap(y1, y2);
    }
    int dx = x2 - x1, dy = y2 - y1;
    for (int x = x1; x < x2; x++) {
        int y = y1 + dy*(x-x1)/dx;
        if (img.channels == 3) {
            img.set_pixel(x, y, 0, 0.f);
            img.set_pixel(x, y, 1, 1.f);
            img.set_pixel(x, y, 2, 0.f);
        } else {
            img.set_pixel(x, y, 0, 1.f);
        }
    }
}
