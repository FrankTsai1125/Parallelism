#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <array>
#include <tuple>
#include <cassert>
#include <cstring>
#include <omp.h>

#include "sift.hpp"
#include "image.hpp"



ScaleSpacePyramid generate_gaussian_pyramid(const Image& img, float sigma_min,
                                            int num_octaves, int scales_per_octave)
{
    // assume initial sigma is 1.0 (after resizing) and smooth
    // the image with sigma_diff to reach requried base_sigma
    float base_sigma = sigma_min / MIN_PIX_DIST;
    Image base_img = img.resize(img.width*2, img.height*2, Interpolation::BILINEAR);
    float sigma_diff = std::sqrt(base_sigma*base_sigma - 1.0f);
    
    // Pre-allocate tmp buffer for gaussian_blur reuse (memory optimization)
    // Use the largest size (base_img size) to cover all octaves
    Image tmp_buffer(base_img.width, base_img.height, 1);
    
    base_img = gaussian_blur(base_img, sigma_diff, &tmp_buffer);

    int imgs_per_octave = scales_per_octave + 3;

    // determine sigma values for bluring
    float k = std::pow(2, 1.0/scales_per_octave);
    std::vector<float> sigma_vals {base_sigma};
    for (int i = 1; i < imgs_per_octave; i++) {
        float sigma_prev = base_sigma * std::pow(k, i-1);
        float sigma_total = k * sigma_prev;
        sigma_vals.push_back(std::sqrt(sigma_total*sigma_total - sigma_prev*sigma_prev));
    }

    // create a scale space pyramid of gaussian images
    // images in each octave are half the size of images in the previous one
    ScaleSpacePyramid pyramid = {
        num_octaves,
        imgs_per_octave,
        std::vector<std::vector<Image>>(num_octaves)
    };
    for (int i = 0; i < num_octaves; i++) {
        pyramid.octaves[i].reserve(imgs_per_octave);
        pyramid.octaves[i].push_back(std::move(base_img));
        for (int j = 1; j < sigma_vals.size(); j++) {
            const Image& prev_img = pyramid.octaves[i].back();
            // Use memory-optimized version with tmp buffer reuse
            pyramid.octaves[i].push_back(gaussian_blur(prev_img, sigma_vals[j], &tmp_buffer));
        }
        // prepare base image for next octave
        if (i < num_octaves - 1) {
            const Image& next_base_img = pyramid.octaves[i][imgs_per_octave-3];
            base_img = next_base_img.resize(next_base_img.width/2, next_base_img.height/2,
                                            Interpolation::NEAREST);
        }
    }
    return pyramid;
}

// generate pyramid of difference of gaussians (DoG) images
ScaleSpacePyramid generate_dog_pyramid(const ScaleSpacePyramid& img_pyramid)
{
    ScaleSpacePyramid dog_pyramid = {
        img_pyramid.num_octaves,
        img_pyramid.imgs_per_octave - 1,
        std::vector<std::vector<Image>>(img_pyramid.num_octaves)
    };
    
    // Parallel over octaves
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < dog_pyramid.num_octaves; i++) {
        dog_pyramid.octaves[i].reserve(dog_pyramid.imgs_per_octave);
        for (int j = 1; j < img_pyramid.imgs_per_octave; j++) {
            // Create new image instead of copying - eliminates ~24MB of unnecessary copies
            int width = img_pyramid.octaves[i][j].width;
            int height = img_pyramid.octaves[i][j].height;
            Image diff(width, height, 1);
            
            // Compute difference directly
            const float* src_curr = img_pyramid.octaves[i][j].data;
            const float* src_prev = img_pyramid.octaves[i][j-1].data;
            float* dst = diff.data;
            
            // Parallel over pixels
            #pragma omp simd
            for (int pix_idx = 0; pix_idx < diff.size; pix_idx++) {
                dst[pix_idx] = src_curr[pix_idx] - src_prev[pix_idx];
            }
            
            dog_pyramid.octaves[i].push_back(std::move(diff));
        }
    }
    return dog_pyramid;
}

bool point_is_extremum(const std::vector<Image>& octave, int scale, int x, int y)
{
    const Image& img = octave[scale];
    const Image& prev = octave[scale-1];
    const Image& next = octave[scale+1];

    bool is_min = true, is_max = true;
    float val = img.get_pixel(x, y, 0), neighbor;

    for (int dx : {-1,0,1}) {
        for (int dy : {-1,0,1}) {
            neighbor = prev.get_pixel(x+dx, y+dy, 0);
            if (neighbor > val) is_max = false;
            if (neighbor < val) is_min = false;

            neighbor = next.get_pixel(x+dx, y+dy, 0);
            if (neighbor > val) is_max = false;
            if (neighbor < val) is_min = false;

            neighbor = img.get_pixel(x+dx, y+dy, 0);
            if (neighbor > val) is_max = false;
            if (neighbor < val) is_min = false;

            if (!is_min && !is_max) return false;
        }
    }
    return true;
}

// fit a quadratic near the discrete extremum,
// update the keypoint (interpolated) extremum value
// and return offsets of the interpolated extremum from the discrete extremum
std::tuple<float, float, float> fit_quadratic(Keypoint& kp,
                                              const std::vector<Image>& octave,
                                              int scale)
{
    const Image& img = octave[scale];
    const Image& prev = octave[scale-1];
    const Image& next = octave[scale+1];

    float g1, g2, g3;
    float h11, h12, h13, h22, h23, h33;
    int x = kp.i, y = kp.j;

    // gradient 
    g1 = (next.get_pixel(x, y, 0) - prev.get_pixel(x, y, 0)) * 0.5;
    g2 = (img.get_pixel(x+1, y, 0) - img.get_pixel(x-1, y, 0)) * 0.5;
    g3 = (img.get_pixel(x, y+1, 0) - img.get_pixel(x, y-1, 0)) * 0.5;

    // hessian
    h11 = next.get_pixel(x, y, 0) + prev.get_pixel(x, y, 0) - 2*img.get_pixel(x, y, 0);
    h22 = img.get_pixel(x+1, y, 0) + img.get_pixel(x-1, y, 0) - 2*img.get_pixel(x, y, 0);
    h33 = img.get_pixel(x, y+1, 0) + img.get_pixel(x, y-1, 0) - 2*img.get_pixel(x, y, 0);
    h12 = (next.get_pixel(x+1, y, 0) - next.get_pixel(x-1, y, 0)
          -prev.get_pixel(x+1, y, 0) + prev.get_pixel(x-1, y, 0)) * 0.25;
    h13 = (next.get_pixel(x, y+1, 0) - next.get_pixel(x, y-1, 0)
          -prev.get_pixel(x, y+1, 0) + prev.get_pixel(x, y-1, 0)) * 0.25;
    h23 = (img.get_pixel(x+1, y+1, 0) - img.get_pixel(x+1, y-1, 0)
          -img.get_pixel(x-1, y+1, 0) + img.get_pixel(x-1, y-1, 0)) * 0.25;
    
    // invert hessian
    float hinv11, hinv12, hinv13, hinv22, hinv23, hinv33;
    float det = h11*h22*h33 - h11*h23*h23 - h12*h12*h33 + 2*h12*h13*h23 - h13*h13*h22;
    hinv11 = (h22*h33 - h23*h23) / det;
    hinv12 = (h13*h23 - h12*h33) / det;
    hinv13 = (h12*h23 - h13*h22) / det;
    hinv22 = (h11*h33 - h13*h13) / det;
    hinv23 = (h12*h13 - h11*h23) / det;
    hinv33 = (h11*h22 - h12*h12) / det;

    // find offsets of the interpolated extremum from the discrete extremum
    float offset_s = -hinv11*g1 - hinv12*g2 - hinv13*g3;
    float offset_x = -hinv12*g1 - hinv22*g2 - hinv23*g3;
    float offset_y = -hinv13*g1 - hinv23*g3 - hinv33*g3;

    float interpolated_extrema_val = img.get_pixel(x, y, 0)
                                   + 0.5*(g1*offset_s + g2*offset_x + g3*offset_y);
    kp.extremum_val = interpolated_extrema_val;
    return {offset_s, offset_x, offset_y};
}

bool point_is_on_edge(const Keypoint& kp, const std::vector<Image>& octave, float edge_thresh=C_EDGE)
{
    const Image& img = octave[kp.scale];
    float h11, h12, h22;
    int x = kp.i, y = kp.j;
    h11 = img.get_pixel(x+1, y, 0) + img.get_pixel(x-1, y, 0) - 2*img.get_pixel(x, y, 0);
    h22 = img.get_pixel(x, y+1, 0) + img.get_pixel(x, y-1, 0) - 2*img.get_pixel(x, y, 0);
    h12 = (img.get_pixel(x+1, y+1, 0) - img.get_pixel(x+1, y-1, 0)
          -img.get_pixel(x-1, y+1, 0) + img.get_pixel(x-1, y-1, 0)) * 0.25;

    float det_hessian = h11*h22 - h12*h12;
    float tr_hessian = h11 + h22;
    float edgeness = tr_hessian*tr_hessian / det_hessian;

    if (edgeness > std::pow(edge_thresh+1, 2)/edge_thresh)
        return true;
    else
        return false;
}

void find_input_img_coords(Keypoint& kp, float offset_s, float offset_x, float offset_y,
                                   float sigma_min=SIGMA_MIN,
                                   float min_pix_dist=MIN_PIX_DIST, int n_spo=N_SPO)
{
    kp.sigma = std::pow(2, kp.octave) * sigma_min * std::pow(2, (offset_s+kp.scale)/n_spo);
    kp.x = min_pix_dist * std::pow(2, kp.octave) * (offset_x+kp.i);
    kp.y = min_pix_dist * std::pow(2, kp.octave) * (offset_y+kp.j);
}

bool refine_or_discard_keypoint(Keypoint& kp, const std::vector<Image>& octave,
                                float contrast_thresh, float edge_thresh)
{
    int k = 0;
    bool kp_is_valid = false; 
    while (k++ < MAX_REFINEMENT_ITERS) {
        auto [offset_s, offset_x, offset_y] = fit_quadratic(kp, octave, kp.scale);

        float max_offset = std::max({std::abs(offset_s),
                                     std::abs(offset_x),
                                     std::abs(offset_y)});
        // find nearest discrete coordinates
        kp.scale += std::round(offset_s);
        kp.i += std::round(offset_x);
        kp.j += std::round(offset_y);
        if (kp.scale >= octave.size()-1 || kp.scale < 1)
            break;

        bool valid_contrast = std::abs(kp.extremum_val) > contrast_thresh;
        if (max_offset < 0.6 && valid_contrast && !point_is_on_edge(kp, octave, edge_thresh)) {
            find_input_img_coords(kp, offset_s, offset_x, offset_y);
            kp_is_valid = true;
            break;
        }
    }
    return kp_is_valid;
}

std::vector<Keypoint> find_keypoints(const ScaleSpacePyramid& dog_pyramid, float contrast_thresh,
                                     float edge_thresh)
{
    std::vector<Keypoint> keypoints;
    
    // Single-phase: Detect, refine, and collect in one pass
    // Eliminates intermediate candidates vector and reduces memory allocation
    #pragma omp parallel
    {
        std::vector<Keypoint> local_keypoints;
        local_keypoints.reserve(500); // Pre-allocate to reduce reallocation
        
        #pragma omp for collapse(2) schedule(dynamic, 1) nowait
        for (int i = 0; i < dog_pyramid.num_octaves; i++) {
            for (int j = 1; j < dog_pyramid.imgs_per_octave-1; j++) {
                const Image& img = dog_pyramid.octaves[i][j];
                const int width = img.width;
                const int height = img.height;
                const float* img_data = img.data;
                const float thresh = 0.8f * contrast_thresh;
                
                // Cache-friendly loop order: iterate y in outer loop for better spatial locality
                for (int y = 1; y < height-1; y++) {
                    for (int x = 1; x < width-1; x++) {
                        const float val = img_data[y * width + x];
                        
                        // Early rejection: contrast threshold check
                        if (std::abs(val) < thresh) {
                            continue;
                        }
                        
                        // Check if extremum
                        if (point_is_extremum(dog_pyramid.octaves[i], j, x, y)) {
                            Keypoint kp = {x, y, i, j, -1, -1, -1, -1};
                            
                            // Refine immediately, no intermediate storage
                            if (refine_or_discard_keypoint(kp, dog_pyramid.octaves[i],
                                                          contrast_thresh, edge_thresh)) {
                                local_keypoints.push_back(kp);
                            }
                        }
                    }
                }
            }
        }
        
        // Reduce critical section contention by collecting once per thread
        #pragma omp critical
        {
            keypoints.insert(keypoints.end(),
                           local_keypoints.begin(),
                           local_keypoints.end());
        }
    }
    
    return keypoints;
}

// calculate x and y derivatives for all images in the input pyramid
ScaleSpacePyramid generate_gradient_pyramid(const ScaleSpacePyramid& pyramid)
{
    ScaleSpacePyramid grad_pyramid = {
        pyramid.num_octaves,
        pyramid.imgs_per_octave,
        std::vector<std::vector<Image>>(pyramid.num_octaves)
    };
    
    // Pre-allocate structure
    for (int i = 0; i < pyramid.num_octaves; i++) {
        grad_pyramid.octaves[i].resize(pyramid.imgs_per_octave);
    }
    
    // Parallel over all octave-scale combinations with direct indexing
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < pyramid.num_octaves; i++) {
        for (int j = 0; j < pyramid.imgs_per_octave; j++) {
            const int width = pyramid.octaves[i][j].width;
            const int height = pyramid.octaves[i][j].height;
            const float* src_data = pyramid.octaves[i][j].data;
            
            grad_pyramid.octaves[i][j] = Image(width, height, 2);
            float* grad_data = grad_pyramid.octaves[i][j].data;
            float* gx_data = grad_data;
            float* gy_data = grad_data + width * height;
            
            // Cache-friendly loop order: y in outer loop for row-wise access
            for (int y = 1; y < height-1; y++) {
                const int row_offset = y * width;
                const int row_above = (y-1) * width;
                const int row_below = (y+1) * width;
                
                // Process entire row at once for better cache locality
                for (int x = 1; x < width-1; x++) {
                    const int idx = row_offset + x;
                    // gx channel: horizontal gradient
                    gx_data[idx] = (src_data[row_offset + x+1] - src_data[row_offset + x-1]) * 0.5f;
                    // gy channel: vertical gradient
                    gy_data[idx] = (src_data[row_below + x] - src_data[row_above + x]) * 0.5f;
                }
            }
        }
    }
    
    return grad_pyramid;
}

// convolve 6x with box filter
void smooth_histogram(float hist[N_BINS])
{
    float tmp_hist[N_BINS];
    const float inv3 = 1.0f / 3.0f;
    
    // Ping-pong between hist and tmp_hist to reduce memory copying
    for (int i = 0; i < 6; i++) {
        float* src = (i % 2 == 0) ? hist : tmp_hist;
        float* dst = (i % 2 == 0) ? tmp_hist : hist;
        
        // Unrolled loop for better performance
        for (int j = 0; j < N_BINS; j++) {
            int prev_idx = (j == 0) ? N_BINS-1 : j-1;
            int next_idx = (j == N_BINS-1) ? 0 : j+1;
            dst[j] = (src[prev_idx] + src[j] + src[next_idx]) * inv3;
        }
    }
    
    // Ensure result is in hist
    if (6 % 2 == 1) {
        for (int j = 0; j < N_BINS; j++) {
            hist[j] = tmp_hist[j];
        }
    }
}

std::vector<float> find_keypoint_orientations(Keypoint& kp, 
                                              const ScaleSpacePyramid& grad_pyramid,
                                              float lambda_ori, float lambda_desc)
{
    float pix_dist = MIN_PIX_DIST * std::pow(2, kp.octave);
    const Image& img_grad = grad_pyramid.octaves[kp.octave][kp.scale];

    // discard kp if too close to image borders 
    float min_dist_from_border = std::min({kp.x, kp.y, pix_dist*img_grad.width-kp.x,
                                           pix_dist*img_grad.height-kp.y});
    if (min_dist_from_border <= std::sqrt(2)*lambda_desc*kp.sigma) {
        return {};
    }

    float hist[N_BINS] = {};
    int bin;
    float gx, gy, grad_norm, weight, theta;
    float patch_sigma = lambda_ori * kp.sigma;
    float patch_radius = 3 * patch_sigma;
    int x_start = std::round((kp.x - patch_radius)/pix_dist);
    int x_end = std::round((kp.x + patch_radius)/pix_dist);
    int y_start = std::round((kp.y - patch_radius)/pix_dist);
    int y_end = std::round((kp.y + patch_radius)/pix_dist);

    // accumulate gradients in orientation histogram
    for (int x = x_start; x <= x_end; x++) {
        for (int y = y_start; y <= y_end; y++) {
            gx = img_grad.get_pixel(x, y, 0);
            gy = img_grad.get_pixel(x, y, 1);
            grad_norm = std::sqrt(gx*gx + gy*gy);
            weight = std::exp(-(std::pow(x*pix_dist-kp.x, 2)+std::pow(y*pix_dist-kp.y, 2))
                              /(2*patch_sigma*patch_sigma));
            theta = std::fmod(std::atan2(gy, gx)+2*M_PI, 2*M_PI);
            bin = (int)std::round(N_BINS/(2*M_PI)*theta) % N_BINS;
            hist[bin] += weight * grad_norm;
        }
    }

    smooth_histogram(hist);

    // extract reference orientations
    float ori_thresh = 0.8, ori_max = 0;
    std::vector<float> orientations;
    for (int j = 0; j < N_BINS; j++) {
        if (hist[j] > ori_max) {
            ori_max = hist[j];
        }
    }
    for (int j = 0; j < N_BINS; j++) {
        if (hist[j] >= ori_thresh * ori_max) {
            float prev = hist[(j-1+N_BINS)%N_BINS], next = hist[(j+1)%N_BINS];
            if (prev > hist[j] || next > hist[j])
                continue;
            float theta = 2*M_PI*(j+1)/N_BINS + M_PI/N_BINS*(prev-next)/(prev-2*hist[j]+next);
            orientations.push_back(theta);
        }
    }
    return orientations;
}

void update_histograms(float hist[N_HIST][N_HIST][N_ORI], float x, float y,
                       float contrib, float theta_mn, float lambda_desc)
{
    float x_i, y_j;
    for (int i = 1; i <= N_HIST; i++) {
        x_i = (i-(1+(float)N_HIST)/2) * 2*lambda_desc/N_HIST;
        if (std::abs(x_i-x) > 2*lambda_desc/N_HIST)
            continue;
        for (int j = 1; j <= N_HIST; j++) {
            y_j = (j-(1+(float)N_HIST)/2) * 2*lambda_desc/N_HIST;
            if (std::abs(y_j-y) > 2*lambda_desc/N_HIST)
                continue;
            
            float hist_weight = (1 - N_HIST*0.5/lambda_desc*std::abs(x_i-x))
                               *(1 - N_HIST*0.5/lambda_desc*std::abs(y_j-y));

            for (int k = 1; k <= N_ORI; k++) {
                float theta_k = 2*M_PI*(k-1)/N_ORI;
                float theta_diff = std::fmod(theta_k-theta_mn+2*M_PI, 2*M_PI);
                if (std::abs(theta_diff) >= 2*M_PI/N_ORI)
                    continue;
                float bin_weight = 1 - N_ORI*0.5/M_PI*std::abs(theta_diff);
                hist[i-1][j-1][k-1] += hist_weight*bin_weight*contrib;
            }
        }
    }
}

void hists_to_vec(float histograms[N_HIST][N_HIST][N_ORI], std::array<uint8_t, 128>& feature_vec)
{
    constexpr int size = N_HIST*N_HIST*N_ORI;
    float *hist = reinterpret_cast<float *>(histograms);

    // Compute L2 norm with SIMD hint
    float norm = 0;
    #pragma omp simd reduction(+:norm)
    for (int i = 0; i < size; i++) {
        norm += hist[i] * hist[i];
    }
    norm = std::sqrt(norm);
    
    // Clamp and compute second norm
    const float thresh = 0.2f * norm;
    float norm2 = 0;
    #pragma omp simd reduction(+:norm2)
    for (int i = 0; i < size; i++) {
        hist[i] = std::min(hist[i], thresh);
        norm2 += hist[i] * hist[i];
    }
    norm2 = std::sqrt(norm2);
    
    // Quantize to uint8
    const float scale = 512.0f / norm2;
    for (int i = 0; i < size; i++) {
        int val = static_cast<int>(hist[i] * scale);
        feature_vec[i] = static_cast<uint8_t>(std::min(val, 255));
    }
}

void compute_keypoint_descriptor(Keypoint& kp, float theta,
                                 const ScaleSpacePyramid& grad_pyramid,
                                 float lambda_desc)
{
    float pix_dist = MIN_PIX_DIST * std::pow(2, kp.octave);
    const Image& img_grad = grad_pyramid.octaves[kp.octave][kp.scale];
    float histograms[N_HIST][N_HIST][N_ORI] = {0};

    //find start and end coords for loops over image patch
    float half_size = std::sqrt(2)*lambda_desc*kp.sigma*(N_HIST+1.)/N_HIST;
    int x_start = std::round((kp.x-half_size) / pix_dist);
    int x_end = std::round((kp.x+half_size) / pix_dist);
    int y_start = std::round((kp.y-half_size) / pix_dist);
    int y_end = std::round((kp.y+half_size) / pix_dist);

    float cos_t = std::cos(theta), sin_t = std::sin(theta);
    float patch_sigma = lambda_desc * kp.sigma;
    //accumulate samples into histograms
    for (int m = x_start; m <= x_end; m++) {
        for (int n = y_start; n <= y_end; n++) {
            // find normalized coords w.r.t. kp position and reference orientation
            float x = ((m*pix_dist - kp.x)*cos_t
                      +(n*pix_dist - kp.y)*sin_t) / kp.sigma;
            float y = (-(m*pix_dist - kp.x)*sin_t
                       +(n*pix_dist - kp.y)*cos_t) / kp.sigma;

            // verify (x, y) is inside the description patch
            if (std::max(std::abs(x), std::abs(y)) > lambda_desc*(N_HIST+1.)/N_HIST)
                continue;

            float gx = img_grad.get_pixel(m, n, 0), gy = img_grad.get_pixel(m, n, 1);
            float theta_mn = std::fmod(std::atan2(gy, gx)-theta+4*M_PI, 2*M_PI);
            float grad_norm = std::sqrt(gx*gx + gy*gy);
            float weight = std::exp(-(std::pow(m*pix_dist-kp.x, 2)+std::pow(n*pix_dist-kp.y, 2))
                                    /(2*patch_sigma*patch_sigma));
            float contribution = weight * grad_norm;

            update_histograms(histograms, x, y, contribution, theta_mn, lambda_desc);
        }
    }

    // build feature vector (descriptor) from histograms
    hists_to_vec(histograms, kp.descriptor);
}

std::vector<Keypoint> find_keypoints_and_descriptors(const Image& img, float sigma_min,
                                                     int num_octaves, int scales_per_octave, 
                                                     float contrast_thresh, float edge_thresh, 
                                                     float lambda_ori, float lambda_desc)
{
    assert(img.channels == 1 || img.channels == 3);

    const Image& input = img.channels == 1 ? img : rgb_to_grayscale(img);
    ScaleSpacePyramid gaussian_pyramid = generate_gaussian_pyramid(input, sigma_min, num_octaves,
                                                                   scales_per_octave);
    ScaleSpacePyramid dog_pyramid = generate_dog_pyramid(gaussian_pyramid);
    std::vector<Keypoint> tmp_kps = find_keypoints(dog_pyramid, contrast_thresh, edge_thresh);
    ScaleSpacePyramid grad_pyramid = generate_gradient_pyramid(gaussian_pyramid);
    
    std::vector<Keypoint> kps;
    kps.reserve(tmp_kps.size() * 2); // Pre-allocate: typically 1-2 orientations per keypoint

    // Parallel descriptor computation with reduced critical section
    #pragma omp parallel
    {
        std::vector<Keypoint> local_kps;
        local_kps.reserve(tmp_kps.size() / omp_get_num_threads() + 10);
        
        #pragma omp for schedule(dynamic, 16) nowait
        for (size_t i = 0; i < tmp_kps.size(); i++) {
            Keypoint& kp_tmp = tmp_kps[i];
            std::vector<float> orientations = find_keypoint_orientations(kp_tmp, grad_pyramid,
                                                                         lambda_ori, lambda_desc);
            for (float theta : orientations) {
                Keypoint kp = kp_tmp;
                compute_keypoint_descriptor(kp, theta, grad_pyramid, lambda_desc);
                local_kps.push_back(kp);
            }
        }
        
        #pragma omp critical
        {
            kps.insert(kps.end(), local_kps.begin(), local_kps.end());
        }
    }
    
    return kps;
}

float euclidean_dist(std::array<uint8_t, 128>& a, std::array<uint8_t, 128>& b)
{
    float dist = 0;
    for (int i = 0; i < 128; i++) {
        int di = (int)a[i] - b[i];
        dist += di * di;
    }
    return std::sqrt(dist);
}

Image draw_keypoints(const Image& img, const std::vector<Keypoint>& kps)
{
    Image res(img);
    if (img.channels == 1) {
        res = grayscale_to_rgb(res);
    }
    for (auto& kp : kps) {
        draw_point(res, kp.x, kp.y, 5);
    }
    return res;
}

// MPI helper functions implementation
void mpi_broadcast_image(Image& img, int root, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    
    // Broadcast image dimensions
    int dims[3];
    if (rank == root) {
        dims[0] = img.width;
        dims[1] = img.height;
        dims[2] = img.channels;
    }
    MPI_Bcast(dims, 3, MPI_INT, root, comm);
    
    // Allocate image on non-root ranks
    if (rank != root) {
        img = Image(dims[0], dims[1], dims[2]);
    }
    
    // Broadcast image data
    MPI_Bcast(img.data, img.size, MPI_FLOAT, root, comm);
}

void compute_octave_partition(int total_octaves, int world_size,
                              std::vector<int>& octave_starts,
                              std::vector<int>& octave_counts) {
    octave_starts.resize(world_size);
    octave_counts.resize(world_size);
    
    if (world_size == 1) {
        // Single process: handle all octaves
        octave_starts[0] = 0;
        octave_counts[0] = total_octaves;
        return;
    }
    
    // Workload-aware strategy for multi-process:
    // Octave 0 is huge (~75% of work), rank 0 handles it alone
    // Remaining octaves 1..N-1 distributed among ranks 1..n-1
    // This way rank 0 does ~75% and other ranks share ~25%
    
    if (world_size == 2) {
        // Special case: 2 processes
        // Rank 0: octave 0 (75%)
        // Rank 1: octaves 1-7 (25%)
        octave_starts[0] = 0;
        octave_counts[0] = 1;
        octave_starts[1] = 1;
        octave_counts[1] = total_octaves - 1;
    } else {
        // General case: n >= 3 processes
        // Rank 0: octave 0 alone
        octave_starts[0] = 0;
        octave_counts[0] = 1;
        
        // Remaining octaves distributed among ranks 1..n-1
        int remaining_octaves = total_octaves - 1;
        int remaining_ranks = world_size - 1;
        int base_count = remaining_octaves / remaining_ranks;
        int remainder = remaining_octaves % remaining_ranks;
        
        int current_octave = 1;
        for (int rank = 1; rank < world_size; ++rank) {
            octave_starts[rank] = current_octave;
            octave_counts[rank] = base_count + (rank - 1 < remainder ? 1 : 0);
            current_octave += octave_counts[rank];
        }
    }
}

std::vector<Keypoint> mpi_gather_keypoints(const std::vector<Keypoint>& local_kps,
                                           int root, MPI_Comm comm) {
    int world_size, rank;
    MPI_Comm_size(comm, &world_size);
    MPI_Comm_rank(comm, &rank);
    
    // Gather counts from all ranks
    int local_count = static_cast<int>(local_kps.size());
    std::vector<int> all_counts(world_size);
    MPI_Gather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, root, comm);
    
    // Prepare buffer for gathering
    std::vector<int> displs(world_size);
    int total_count = 0;
    if (rank == root) {
        for (int i = 0; i < world_size; ++i) {
            displs[i] = total_count;
            total_count += all_counts[i];
        }
    }
    
    // Pack keypoint data into a flat structure
    struct FlatKeypoint {
        int i, j, octave, scale;
        float x, y, sigma, extremum_val;
        uint8_t descriptor[128];
    };
    
    // Convert local keypoints to flat format
    std::vector<FlatKeypoint> flat_local(local_count);
    for (int i = 0; i < local_count; ++i) {
        flat_local[i].i = local_kps[i].i;
        flat_local[i].j = local_kps[i].j;
        flat_local[i].octave = local_kps[i].octave;
        flat_local[i].scale = local_kps[i].scale;
        flat_local[i].x = local_kps[i].x;
        flat_local[i].y = local_kps[i].y;
        flat_local[i].sigma = local_kps[i].sigma;
        flat_local[i].extremum_val = local_kps[i].extremum_val;
        memcpy(flat_local[i].descriptor, local_kps[i].descriptor.data(), 128);
    }
    
    // Gather all keypoints
    std::vector<FlatKeypoint> flat_all;
    if (rank == root) {
        flat_all.resize(total_count);
    }
    
    // Use MPI_Gatherv with byte transfer
    int local_bytes = local_count * sizeof(FlatKeypoint);
    std::vector<int> byte_counts(world_size);
    std::vector<int> byte_displs(world_size);
    
    if (rank == root) {
        for (int i = 0; i < world_size; ++i) {
            byte_counts[i] = all_counts[i] * sizeof(FlatKeypoint);
            byte_displs[i] = displs[i] * sizeof(FlatKeypoint);
        }
    }
    
    MPI_Gatherv(flat_local.data(), local_bytes, MPI_BYTE,
                flat_all.data(), byte_counts.data(), byte_displs.data(), MPI_BYTE,
                root, comm);
    
    // Convert back to Keypoint vector
    std::vector<Keypoint> result;
    if (rank == root) {
        result.reserve(total_count);
        for (const auto& fkp : flat_all) {
            Keypoint kp;
            kp.i = fkp.i;
            kp.j = fkp.j;
            kp.octave = fkp.octave;
            kp.scale = fkp.scale;
            kp.x = fkp.x;
            kp.y = fkp.y;
            kp.sigma = fkp.sigma;
            kp.extremum_val = fkp.extremum_val;
            memcpy(kp.descriptor.data(), fkp.descriptor, 128);
            result.push_back(kp);
        }
    }
    
    return result;
}

// Process a specific range of octaves for MPI distribution
std::vector<Keypoint> find_keypoints_range(const ScaleSpacePyramid& dog_pyramid,
                                          const ScaleSpacePyramid& grad_pyramid,
                                          int start_octave, int num_octaves,
                                          float contrast_thresh,
                                          float edge_thresh,
                                          float lambda_ori,
                                          float lambda_desc) {
    std::vector<Keypoint> kps;
    
    // Phase 1: Detect and refine candidates in assigned octaves
    std::vector<Keypoint> tmp_kps;
    
    #pragma omp parallel
    {
        std::vector<Keypoint> thread_local_kps;
        thread_local_kps.reserve(1000); // Pre-allocate
        
        #pragma omp for collapse(2) schedule(dynamic, 1) nowait
        for (int oct = start_octave; oct < start_octave + num_octaves; oct++) {
            for (int scale = 1; scale < dog_pyramid.imgs_per_octave - 1; scale++) {
                const Image& img = dog_pyramid.octaves[oct][scale];
                const int width = img.width;
                const int height = img.height;
                const float* img_data = img.data;
                const float thresh = 0.8f * contrast_thresh;
                
                // Cache-friendly loop order
                for (int y = 1; y < height - 1; y++) {
                    for (int x = 1; x < width - 1; x++) {
                        const float val = img_data[y * width + x];
                        
                        if (std::abs(val) < thresh) {
                            continue;
                        }
                        
                        if (point_is_extremum(dog_pyramid.octaves[oct], scale, x, y)) {
                            Keypoint kp = {x, y, oct, scale, -1, -1, -1, -1};
                            if (refine_or_discard_keypoint(kp, dog_pyramid.octaves[oct], 
                                                          contrast_thresh, edge_thresh)) {
                                thread_local_kps.push_back(kp);
                            }
                        }
                    }
                }
            }
        }
        
        #pragma omp critical
        {
            tmp_kps.insert(tmp_kps.end(), thread_local_kps.begin(), thread_local_kps.end());
        }
    }
    
    // Phase 2: Compute descriptors with pre-allocation
    kps.reserve(tmp_kps.size() * 2);
    
    #pragma omp parallel
    {
        std::vector<Keypoint> local_kps;
        local_kps.reserve(tmp_kps.size() / omp_get_num_threads() + 10);
        
        #pragma omp for schedule(dynamic, 16) nowait
        for (size_t i = 0; i < tmp_kps.size(); i++) {
            Keypoint kp = tmp_kps[i];
            std::vector<float> orientations = find_keypoint_orientations(
                kp, grad_pyramid, lambda_ori, lambda_desc);
            
            for (float theta : orientations) {
                Keypoint final_kp = kp;
                compute_keypoint_descriptor(final_kp, theta, grad_pyramid, lambda_desc);
                local_kps.push_back(final_kp);
            }
        }
        
        #pragma omp critical
        {
            kps.insert(kps.end(), local_kps.begin(), local_kps.end());
        }
    }
    
    return kps;
}