#include <hip/hip_runtime.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

namespace param {
    const int n_steps = 200000;
    const double dt = 60;
    const double eps = 1e-3;
    const double eps2 = eps * eps;
    const double G = 6.674e-11;
    const double planet_radius = 1e7;
    const double missile_speed = 1e6;

    __device__ __host__ inline double gravity_device_mass(double m0, double t) {
        return m0 + 0.5 * m0 * fabs(sin(t / 6000.0));
    }

    __device__ __host__ inline double get_missile_cost(double t) { 
        return 1e5 + 1e3 * t; 
    }
}

void read_input(const char* filename, int& n, int& planet, int& asteroid,
    std::vector<double>& qx, std::vector<double>& qy, std::vector<double>& qz,
    std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz,
    std::vector<double>& m, std::vector<int>& type) {
    std::ifstream fin(filename);
    fin >> n >> planet >> asteroid;
    qx.resize(n); qy.resize(n); qz.resize(n);
    vx.resize(n); vy.resize(n); vz.resize(n);
    m.resize(n); type.resize(n);
    
    for (int i = 0; i < n; i++) {
        std::string type_str;
        fin >> qx[i] >> qy[i] >> qz[i] >> vx[i] >> vy[i] >> vz[i] >> m[i] >> type_str;
        type[i] = (type_str == "device") ? 1 : 0;
    }
}

void write_output(const char* filename, double min_dist, int hit_time_step,
    int gravity_device_id, double missile_cost) {
    std::ofstream fout(filename);
    fout << std::scientific
         << std::setprecision(std::numeric_limits<double>::digits10 + 1) << min_dist << '\n'
         << hit_time_step << '\n'
         << gravity_device_id << ' ' << missile_cost << '\n';
}

// ========== GPU Kernels ==========

// Strategy 1: Single Block Kernel (Optimized for N <= 256)
__global__ void simulation_single_block_kernel(
    int n, int start_step, int steps_to_run,
    int planet, int asteroid, int missile_target_id,
    double* __restrict__ qx, double* __restrict__ qy, double* __restrict__ qz,
    double* __restrict__ vx, double* __restrict__ vy, double* __restrict__ vz,
    double* __restrict__ m, const int* __restrict__ type,
    double* __restrict__ min_dist_ptr,
    int* __restrict__ hit_step_ptr,
    int* __restrict__ missile_hit_step_ptr,
    int* __restrict__ stop_flag)
{
    if (blockIdx.x > 0) return;

    extern __shared__ double s_mem[];
    double* s_qx = s_mem;        double* s_qy = s_qx + n;
    double* s_qz = s_qy + n;     double* s_vx = s_qz + n;
    double* s_vy = s_vx + n;     double* s_vz = s_vy + n;
    double* s_m  = s_vz + n;
    
    int tid = threadIdx.x;

    // Load to Shared Memory
    if (tid < n) {
        s_qx[tid] = qx[tid]; s_qy[tid] = qy[tid]; s_qz[tid] = qz[tid];
        s_vx[tid] = vx[tid]; s_vy[tid] = vy[tid]; s_vz[tid] = vz[tid];
        s_m[tid]  = m[tid];
    }
    __syncthreads();

    for (int s = 0; s < steps_to_run; ++s) {
        int step = start_step + s;
        // Only thread 0 checks stop flag to minimize memory traffic
        // But we need to broadcast it? No, if thread 0 breaks, others might continue?
        // Actually, we need all threads to exit. 
        // But reading global memory every step is expensive.
        // Let's optimize: only check stop_flag every 100 steps or if we know we wrote to it.
        // For now, keep it simple but minimal.
        if (*stop_flag) break;

        // 1. Update Mass
        if (tid < n) {
            if (type[tid] == 1) { 
                double m0 = m[tid]; 
                double t = (double)(step + 1) * param::dt;
                s_m[tid] = param::gravity_device_mass(m0, t);
            }
            if (missile_target_id >= 0 && *missile_hit_step_ptr != -1 && *missile_hit_step_ptr < (step + 1)) {
                if (tid == missile_target_id) s_m[tid] = 0.0;
            }
        }
        __syncthreads();

        // 2. Force Calculation
        double fx = 0.0, fy = 0.0, fz = 0.0;
        double my_qx, my_qy, my_qz;

        if (tid < n) {
            my_qx = s_qx[tid]; my_qy = s_qy[tid]; my_qz = s_qz[tid];
        }
        
        if (tid < n) {
            // Unroll loop manually for small N? Compiler -O3 usually does it.
            #pragma unroll 4
            for (int j = 0; j < n; ++j) {
                if (tid == j) continue;
                double dx = s_qx[j] - my_qx;
                double dy = s_qy[j] - my_qy;
                double dz = s_qz[j] - my_qz;
                double dist2 = dx*dx + dy*dy + dz*dz + param::eps2;
                double invDist = rsqrt(dist2);
                double f = param::G * s_m[j] * invDist * invDist * invDist;
                fx += f * dx; fy += f * dy; fz += f * dz;
            }
        }
        __syncthreads();

        // 3. Update Motion
        if (tid < n) {
            s_vx[tid] += fx * param::dt;
            s_vy[tid] += fy * param::dt;
            s_vz[tid] += fz * param::dt;

            s_qx[tid] += s_vx[tid] * param::dt;
            s_qy[tid] += s_vy[tid] * param::dt;
            s_qz[tid] += s_vz[tid] * param::dt;
        }
        __syncthreads();

        // 4. Check Status (Thread 0 only)
        if (tid == 0) {
            int check_step = step + 1;
            double dx = s_qx[planet] - s_qx[asteroid];
            double dy = s_qy[planet] - s_qy[asteroid];
            double dz = s_qz[planet] - s_qz[asteroid];
            double dist = sqrt(dx*dx + dy*dy + dz*dz);
            
            if (dist < *min_dist_ptr) *min_dist_ptr = dist;
            
            if (*hit_step_ptr == -2 && dist < param::planet_radius) {
                *hit_step_ptr = check_step;
                *stop_flag = 1;
            }
            
            if (missile_target_id >= 0 && *missile_hit_step_ptr == -1) {
                double m_dx = s_qx[planet] - s_qx[missile_target_id];
                double m_dy = s_qy[planet] - s_qy[missile_target_id];
                double m_dz = s_qz[planet] - s_qz[missile_target_id];
                double m_dist = sqrt(m_dx*m_dx + m_dy*m_dy + m_dz*m_dz);
                
                if ((double)check_step * param::dt * param::missile_speed > m_dist) {
                    *missile_hit_step_ptr = check_step;
                }
            }
        }
        __syncthreads();
    }

    // Write back
    if (tid < n) {
        qx[tid] = s_qx[tid]; qy[tid] = s_qy[tid]; qz[tid] = s_qz[tid];
        vx[tid] = s_vx[tid]; vy[tid] = s_vy[tid]; vz[tid] = s_vz[tid];
    }
}

// Strategy 2: One Block Per Particle (Optimized for N > 256)
__device__ void blockReduce(volatile double* sdata, int tid, int blockSize) {
    __syncthreads();
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64)  { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) {
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
        if (tid < 16) {
            sdata[tid] += sdata[tid + 16]; sdata[tid] += sdata[tid + 8];
            sdata[tid] += sdata[tid + 4];  sdata[tid] += sdata[tid + 2];
            sdata[tid] += sdata[tid + 1];
        }
    }
}

__global__ void update_masses_kernel(
    int n, double t, const double* __restrict__ m0, const int* __restrict__ type,
    double* __restrict__ m_current,
    int missile_target_id, const int* __restrict__ missile_hit_step_ptr)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double m = m0[i];
        if (type[i] == 1) m = param::gravity_device_mass(m, t);
        
        if (missile_target_id >= 0 && *missile_hit_step_ptr != -1) {
             if ((double)(*missile_hit_step_ptr + 1) * param::dt <= t + 1e-9) {
                 if (i == missile_target_id) m = 0.0;
            }
        }
        m_current[i] = m;
    }
}

__global__ void compute_accel_per_particle(
    int n,
    const double* __restrict__ qx, const double* __restrict__ qy, const double* __restrict__ qz,
    const double* __restrict__ m0, const int* __restrict__ type, double t,
    double* __restrict__ ax, double* __restrict__ ay, double* __restrict__ az,
    int missile_target_id, const int* __restrict__ missile_hit_step_ptr) 
{
    int i = blockIdx.x;
    if (i >= n) return;

    double my_qx = qx[i]; double my_qy = qy[i]; double my_qz = qz[i];
    double fx = 0.0, fy = 0.0, fz = 0.0;
    
    int hit_step = (missile_target_id >= 0) ? *missile_hit_step_ptr : -1;

    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        if (i == j) continue;
        double dx = qx[j] - my_qx;
        double dy = qy[j] - my_qy;
        double dz = qz[j] - my_qz;
        double dist2 = dx*dx + dy*dy + dz*dz + param::eps2;
        double invDist = rsqrt(dist2);
        
        double m_val = m0[j];
        if (type[j] == 1) m_val = param::gravity_device_mass(m_val, t);
        
        if (hit_step != -1 && j == missile_target_id) {
            if ((double)(hit_step + 1) * param::dt <= t + 1e-9) {
                m_val = 0.0;
            }
        }
        
        double f = param::G * m_val * invDist * invDist * invDist;
        fx += f * dx; fy += f * dy; fz += f * dz;
    }

    __shared__ double s_fx[256]; __shared__ double s_fy[256]; __shared__ double s_fz[256];
    s_fx[threadIdx.x] = fx; s_fy[threadIdx.x] = fy; s_fz[threadIdx.x] = fz;

    blockReduce(s_fx, threadIdx.x, blockDim.x);
    blockReduce(s_fy, threadIdx.x, blockDim.x);
    blockReduce(s_fz, threadIdx.x, blockDim.x);

    if (threadIdx.x == 0) {
        ax[i] = s_fx[0]; ay[i] = s_fy[0]; az[i] = s_fz[0];
    }
}

__global__ void update_motion_kernel(
    int n, double* vx, double* vy, double* vz,
    double* qx, double* qy, double* qz,
    const double* ax, const double* ay, const double* az,
    int step, int planet, int asteroid,
    double* min_dist_ptr, int* hit_step_ptr,
    int missile_target_id, int* missile_hit_step_ptr, int* stop_flag,
    bool do_check) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        vx[i] += ax[i] * param::dt; vy[i] += ay[i] * param::dt; vz[i] += az[i] * param::dt;
        qx[i] += vx[i] * param::dt; qy[i] += vy[i] * param::dt; qz[i] += vz[i] * param::dt;
    }
    
    if (do_check && blockIdx.x == 0) {
        __syncthreads(); // Wait for Block 0 updates
        if (threadIdx.x == 0) {
            double dx = qx[planet] - qx[asteroid];
            double dy = qy[planet] - qy[asteroid];
            double dz = qz[planet] - qz[asteroid];
            double dist = sqrt(dx*dx + dy*dy + dz*dz);
            
            if (dist < *min_dist_ptr) *min_dist_ptr = dist;
            
            int check_step = step + 1;
            if (*hit_step_ptr == -2 && dist < param::planet_radius) {
                *hit_step_ptr = check_step;
                *stop_flag = 1;
            }
            
            if (missile_target_id >= 0 && *missile_hit_step_ptr == -1) {
                double m_dx = qx[planet] - qx[missile_target_id];
                double m_dy = qy[planet] - qy[missile_target_id];
                double m_dz = qz[planet] - qz[missile_target_id];
                double m_dist = sqrt(m_dx*m_dx + m_dy*m_dy + m_dz*m_dz);
                
                if ((double)check_step * param::dt * param::missile_speed > m_dist) {
                    *missile_hit_step_ptr = check_step;
                }
            }
        }
    }
}

__global__ void check_status_kernel(
    int n, int step, int planet, int asteroid,
    const double* qx, const double* qy, const double* qz,
    double* min_dist_ptr, int* hit_step_ptr,
    int missile_target_id, int* missile_hit_step_ptr, int* stop_flag)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        double dist = sqrt(dx*dx + dy*dy + dz*dz);
        
        if (dist < *min_dist_ptr) *min_dist_ptr = dist;
        
        // Check Step Logic:
        // If we are at loop index 'step', and we just updated motion.
        // The state corresponds to time (step + 1) * dt?
        // Wait, let's be consistent with Strategy 1.
        // Strategy 1: check_step = step + 1.
        // Here we pass 'step'. So we should use step + 1.
        int check_step = step + 1;
        
        if (*hit_step_ptr == -2 && dist < param::planet_radius) {
            *hit_step_ptr = check_step;
            *stop_flag = 1;
        }
        
        if (missile_target_id >= 0 && *missile_hit_step_ptr == -1) {
            double m_dx = qx[planet] - qx[missile_target_id];
            double m_dy = qy[planet] - qy[missile_target_id];
            double m_dz = qz[planet] - qz[missile_target_id];
            double m_dist = sqrt(m_dx*m_dx + m_dy*m_dy + m_dz*m_dz);
            
            if ((double)check_step * param::dt * param::missile_speed > m_dist) {
                *missile_hit_step_ptr = check_step;
            }
        }
    }
}

void run_simulation_gpu(int n, int planet, int asteroid,
    std::vector<double>& qx, std::vector<double>& qy, std::vector<double>& qz,
    std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz,
    const std::vector<double>& m_in, const std::vector<int>& type,
    int max_steps, double& min_dist, int& hit_time_step,
    int missile_target_id = -1, int* missile_hit_step = nullptr) {
    
    double *d_qx, *d_qy, *d_qz, *d_vx, *d_vy, *d_vz, *d_m;
    double *d_ax, *d_ay, *d_az;
    int *d_type;
    double *d_min_dist_val;
    int *d_hit_step_val, *d_missile_hit_val, *d_stop_flag; 
    
    size_t size = n * sizeof(double);
    size_t size_int = n * sizeof(int);
    
    HIP_CHECK(hipMalloc(&d_qx, size)); HIP_CHECK(hipMalloc(&d_qy, size)); HIP_CHECK(hipMalloc(&d_qz, size));
    HIP_CHECK(hipMalloc(&d_vx, size)); HIP_CHECK(hipMalloc(&d_vy, size)); HIP_CHECK(hipMalloc(&d_vz, size));
    HIP_CHECK(hipMalloc(&d_m, size));
    HIP_CHECK(hipMalloc(&d_type, size_int));
    HIP_CHECK(hipMalloc(&d_ax, size)); HIP_CHECK(hipMalloc(&d_ay, size)); HIP_CHECK(hipMalloc(&d_az, size));
    
    HIP_CHECK(hipMalloc(&d_min_dist_val, sizeof(double)));
    HIP_CHECK(hipMalloc(&d_hit_step_val, sizeof(int)));
    HIP_CHECK(hipMalloc(&d_missile_hit_val, sizeof(int)));
    HIP_CHECK(hipMalloc(&d_stop_flag, sizeof(int)));
    
    HIP_CHECK(hipMemcpy(d_qx, qx.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_qy, qy.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_qz, qz.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vx, vx.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vy, vy.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vz, vz.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_m, m_in.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_type, type.data(), size_int, hipMemcpyHostToDevice));
    
    double init_min_dist = std::numeric_limits<double>::infinity();
    int init_hit_step = -2;
    int init_missile_hit = -1;
    int init_stop = 0;
    HIP_CHECK(hipMemcpy(d_min_dist_val, &init_min_dist, sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_hit_step_val, &init_hit_step, sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_missile_hit_val, &init_missile_hit, sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_stop_flag, &init_stop, sizeof(int), hipMemcpyHostToDevice));
    
    if (n <= 64) {
        int blockSize = (n + 31) / 32 * 32;
        if (blockSize < 64) blockSize = 64;
        if (blockSize > 1024) blockSize = 1024;
        
        size_t shared_mem_size = 1024 * 8 * 7; // Allocate for max supported N
        
        // Optimization: Run all steps in one kernel launch
        // The kernel internally handles stop_flag and breaks loop
        hipLaunchKernelGGL(simulation_single_block_kernel, 
                          dim3(1), dim3(blockSize), shared_mem_size, 0,
                          n, 0, max_steps,
                          planet, asteroid, missile_target_id,
                          d_qx, d_qy, d_qz, d_vx, d_vy, d_vz, d_m, d_type, 
                          d_min_dist_val, d_hit_step_val, d_missile_hit_val, d_stop_flag);
    } 
    else {
        const int BATCH_SIZE = 500; 
        int gridN = n;
        int elemBlock = 256;
        int elemGrid = (n + elemBlock - 1) / elemBlock;
        
        bool can_fuse_check = (planet < 256 && asteroid < 256 && (missile_target_id == -1 || missile_target_id < 256));

        for (int step_start = 0; step_start <= max_steps; step_start += BATCH_SIZE) {
            int step_end = std::min(step_start + BATCH_SIZE - 1, max_steps);
            
            for (int step = step_start; step <= step_end; step++) {
                if (step > 0) {
                    // 1. Compute Acceleration (Inline Mass & Hit Logic)
                    hipLaunchKernelGGL(compute_accel_per_particle, 
                                      dim3(gridN), dim3(256), 0, 0,
                                      n, d_qx, d_qy, d_qz, d_m, d_type, (double)step * param::dt,
                                      d_ax, d_ay, d_az,
                                      missile_target_id, d_missile_hit_val);
                                      
                    // 2. Update Motion (with conditional Fused Check)
                    hipLaunchKernelGGL(update_motion_kernel,
                                      dim3(elemGrid), dim3(elemBlock), 0, 0,
                                      n, d_vx, d_vy, d_vz, d_qx, d_qy, d_qz,
                                      d_ax, d_ay, d_az,
                                      step, planet, asteroid, d_min_dist_val, d_hit_step_val,
                                      missile_target_id, d_missile_hit_val, d_stop_flag,
                                      can_fuse_check);
                    
                    // 3. Check Status (Separate Kernel only if not fused)
                    if (!can_fuse_check) {
                        hipLaunchKernelGGL(check_status_kernel,
                                          dim3(1), dim3(1), 0, 0,
                                          n, step, planet, asteroid,
                                          d_qx, d_qy, d_qz,
                                          d_min_dist_val, d_hit_step_val,
                                          missile_target_id, d_missile_hit_val, d_stop_flag);
                    }
                } else {
                    hipLaunchKernelGGL(check_status_kernel,
                                      dim3(1), dim3(1), 0, 0,
                                      n, step, planet, asteroid,
                                      d_qx, d_qy, d_qz,
                                      d_min_dist_val, d_hit_step_val,
                                      missile_target_id, d_missile_hit_val, d_stop_flag);
                }
            }
            
            int stop;
            HIP_CHECK(hipMemcpy(&stop, d_stop_flag, sizeof(int), hipMemcpyDeviceToHost));
            if (stop) break;
        }
    }
    
    HIP_CHECK(hipMemcpy(&min_dist, d_min_dist_val, sizeof(double), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&hit_time_step, d_hit_step_val, sizeof(int), hipMemcpyDeviceToHost));
    
    if (missile_hit_step) {
        HIP_CHECK(hipMemcpy(missile_hit_step, d_missile_hit_val, sizeof(int), hipMemcpyDeviceToHost));
    }
    
    HIP_CHECK(hipFree(d_qx)); HIP_CHECK(hipFree(d_qy)); HIP_CHECK(hipFree(d_qz));
    HIP_CHECK(hipFree(d_vx)); HIP_CHECK(hipFree(d_vy)); HIP_CHECK(hipFree(d_vz));
    HIP_CHECK(hipFree(d_m)); HIP_CHECK(hipFree(d_type));
    HIP_CHECK(hipFree(d_ax)); HIP_CHECK(hipFree(d_ay)); HIP_CHECK(hipFree(d_az));
    HIP_CHECK(hipFree(d_min_dist_val)); HIP_CHECK(hipFree(d_hit_step_val));
    HIP_CHECK(hipFree(d_missile_hit_val)); HIP_CHECK(hipFree(d_stop_flag));
}

int main(int argc, char** argv) {
    if (argc != 3) {
        throw std::runtime_error("must supply 2 arguments");
    }
    
    int n, planet, asteroid;
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<int> type;
    
    int deviceCount = 0;
    if (hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0) {
        return 1;
    }
    
    // Problem 1
    double min_dist = std::numeric_limits<double>::infinity();
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
    
    std::vector<double> m_p1 = m;
    for (int i = 0; i < n; i++) {
        if (type[i] == 1) m_p1[i] = 0;
    }
    
    int dummy_hit = -2;
    run_simulation_gpu(n, planet, asteroid, qx, qy, qz, vx, vy, vz, 
                      m_p1, type, param::n_steps, min_dist, dummy_hit);
    
    // Problem 2
    int hit_time_step = -2;
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
    
    double dummy_dist = std::numeric_limits<double>::infinity();
    run_simulation_gpu(n, planet, asteroid, qx, qy, qz, vx, vy, vz,
                      m, type, param::n_steps, dummy_dist, hit_time_step);
    
    // Problem 3
    int gravity_device_id = -1;
    double missile_cost = 0.0;
    
    if (hit_time_step >= 0) {
        read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
        std::vector<double> base_qx = qx, base_qy = qy, base_qz = qz;
        std::vector<double> base_vx = vx, base_vy = vy, base_vz = vz;
        std::vector<double> base_m = m;
        std::vector<int> base_type = type;
        
        std::vector<int> devices;
        for (int i = 0; i < n; ++i) {
            if (base_type[i] == 1) devices.push_back(i);
        }
        
        if (!devices.empty()) {
            double best_cost = std::numeric_limits<double>::infinity();
            int best_device = -1;
            const double cost_eps = 1e-6;
            
            for (int device_id : devices) {
                std::vector<double> qx_test = base_qx, qy_test = base_qy, qz_test = base_qz;
                std::vector<double> vx_test = base_vx, vy_test = base_vy, vz_test = base_vz;
                std::vector<double> m_test = base_m;
                
                double test_min_dist = std::numeric_limits<double>::infinity();
                int test_hit = -2;
                int missile_hit = -1;
                
                run_simulation_gpu(n, planet, asteroid, qx_test, qy_test, qz_test,
                                  vx_test, vy_test, vz_test, m_test, base_type,
                                  param::n_steps, test_min_dist, test_hit,
                                  device_id, &missile_hit);
                
                if (test_hit == -2 && missile_hit >= 0) {
                    double cost = param::get_missile_cost(missile_hit * param::dt);
                    if (cost < best_cost - cost_eps ||
                        (std::abs(cost - best_cost) <= cost_eps && 
                         (best_device == -1 || device_id < best_device))) {
                        best_cost = cost;
                        best_device = device_id;
                    }
                }
            }
            
            if (best_device != -1) {
                gravity_device_id = best_device;
                missile_cost = best_cost;
            }
        }
    }
    
    write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);
    return 0;
}