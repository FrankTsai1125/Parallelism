// Monte Carlo option pricing (European / Asian) on NVIDIA Tesla V100 (sm_70).
// - Uses cuRAND (Philox) for RNG
// - Each thread simulates independent price paths (embarrassingly parallel)
// - Outputs estimated price + standard error
//
// Build (Linux, CUDA toolkit installed):
//   nvcc -O3 -std=c++17 -arch=sm_70 mc_pricer.cu -o mc_pricer
//
// Run:
//   ./mc_pricer --paths 5000000 --steps 252 --type european
//   ./mc_pricer --paths 5000000 --steps 252 --type asian
//
// Notes from machine.txt:
// - Tesla V100-SXM2-32GB, compute capability 7.0 => use -arch=sm_70
// - CUDA Driver 12.2, runtime in deviceQuery example 11.7 (either is fine for this code)

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// ----------------------------- Utilities -----------------------------

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::cerr << "CUDA error: " << cudaGetErrorString(err__)                 \
                << " at " << __FILE__ << ":" << __LINE__ << "\n";             \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

enum class OptionType { EuropeanCall, AsianArithmeticCall };

struct Params {
  double S0 = 100.0;
  double K = 100.0;
  double r = 0.05;
  double sigma = 0.2;
  double T = 1.0;
  int steps = 252;
  std::int64_t paths = 5'000'000;
  OptionType type = OptionType::EuropeanCall;
  std::uint64_t seed = 1234;
};

static void usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0
      << " [--type european|asian] [--S0 N] [--K N] [--r N] [--sigma N] [--T N]\n"
      << "           [--steps N] [--paths N] [--seed N] [--cpu]\n";
}

static bool parse_args(int argc, char** argv, Params& p, bool& run_cpu) {
  run_cpu = false;
  for (int i = 1; i < argc; ++i) {
    auto next = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << "\n";
        return nullptr;
      }
      return argv[++i];
    };

    if (std::strcmp(argv[i], "--type") == 0) {
      const char* v = next("--type");
      if (!v) return false;
      if (std::strcmp(v, "european") == 0) p.type = OptionType::EuropeanCall;
      else if (std::strcmp(v, "asian") == 0) p.type = OptionType::AsianArithmeticCall;
      else {
        std::cerr << "Unknown --type: " << v << "\n";
        return false;
      }
    } else if (std::strcmp(argv[i], "--S0") == 0) {
      const char* v = next("--S0"); if (!v) return false;
      p.S0 = std::stod(v);
    } else if (std::strcmp(argv[i], "--K") == 0) {
      const char* v = next("--K"); if (!v) return false;
      p.K = std::stod(v);
    } else if (std::strcmp(argv[i], "--r") == 0) {
      const char* v = next("--r"); if (!v) return false;
      p.r = std::stod(v);
    } else if (std::strcmp(argv[i], "--sigma") == 0) {
      const char* v = next("--sigma"); if (!v) return false;
      p.sigma = std::stod(v);
    } else if (std::strcmp(argv[i], "--T") == 0) {
      const char* v = next("--T"); if (!v) return false;
      p.T = std::stod(v);
    } else if (std::strcmp(argv[i], "--steps") == 0) {
      const char* v = next("--steps"); if (!v) return false;
      p.steps = std::stoi(v);
    } else if (std::strcmp(argv[i], "--paths") == 0) {
      const char* v = next("--paths"); if (!v) return false;
      p.paths = std::stoll(v);
    } else if (std::strcmp(argv[i], "--seed") == 0) {
      const char* v = next("--seed"); if (!v) return false;
      p.seed = static_cast<std::uint64_t>(std::stoull(v));
    } else if (std::strcmp(argv[i], "--cpu") == 0) {
      run_cpu = true;
    } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      return false;
    } else {
      std::cerr << "Unknown arg: " << argv[i] << "\n";
      return false;
    }
  }
  if (p.steps <= 0 || p.paths <= 0 || p.T <= 0.0 || p.sigma < 0.0) {
    std::cerr << "Invalid parameters.\n";
    return false;
  }
  return true;
}

// ----------------------------- GPU implementation -----------------------------

__global__ void mc_kernel(
    std::int64_t paths,
    int steps,
    double S0, double K, double r, double sigma, double T,
    int opt_type, // 0: EuropeanCall, 1: AsianArithmeticCall
    std::uint64_t seed,
    double* sum_payoff,
    double* sum_sq_payoff) {

  // Grid-stride over paths
  const std::int64_t tid = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;

  const double dt = T / static_cast<double>(steps);
  const double drift = (r - 0.5 * sigma * sigma) * dt;
  const double vol = sigma * std::sqrt(dt);
  const double disc = std::exp(-r * T);

  // Philox is a good default for parallel RNG (counter-based).
  curandStatePhilox4_32_10_t st;
  curand_init(static_cast<unsigned long long>(seed), /*subsequence=*/static_cast<unsigned long long>(tid),
              /*offset=*/0ULL, &st);

  double local_sum = 0.0;
  double local_sum_sq = 0.0;

  for (std::int64_t path = tid; path < paths; path += stride) {
    double S = S0;
    double avgS = 0.0;

    // Generate normals in batches of 4 for better throughput.
    int j = 0;
    for (; j + 4 <= steps; j += 4) {
      float4 z = curand_normal4(&st);
      // step 0
      S *= std::exp(drift + vol * static_cast<double>(z.x));
      if (opt_type == 1) avgS += S;
      // step 1
      S *= std::exp(drift + vol * static_cast<double>(z.y));
      if (opt_type == 1) avgS += S;
      // step 2
      S *= std::exp(drift + vol * static_cast<double>(z.z));
      if (opt_type == 1) avgS += S;
      // step 3
      S *= std::exp(drift + vol * static_cast<double>(z.w));
      if (opt_type == 1) avgS += S;
    }
    for (; j < steps; ++j) {
      float z = curand_normal(&st);
      S *= std::exp(drift + vol * static_cast<double>(z));
      if (opt_type == 1) avgS += S;
    }

    double payoff = 0.0;
    if (opt_type == 0) {
      // std::max is host-only in many nvcc configs; use device-friendly max.
      payoff = fmax(S - K, 0.0);
    } else {
      const double meanS = avgS / static_cast<double>(steps);
      payoff = fmax(meanS - K, 0.0);
    }
    payoff *= disc;

    local_sum += payoff;
    local_sum_sq += payoff * payoff;
  }

  // Accumulate global sums. atomicAdd(double) is supported on sm_70.
  atomicAdd(sum_payoff, local_sum);
  atomicAdd(sum_sq_payoff, local_sum_sq);
}

struct MCResult {
  double price = 0.0;
  double std_error = 0.0;
  double ms = 0.0;
};

static MCResult run_gpu(const Params& p) {
  double* d_sum = nullptr;
  double* d_sum_sq = nullptr;
  CUDA_CHECK(cudaMalloc(&d_sum, sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_sum_sq, sizeof(double)));
  CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(double)));
  CUDA_CHECK(cudaMemset(d_sum_sq, 0, sizeof(double)));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  int device = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  // V100: 80 SM. A reasonable default is a few blocks per SM.
  const int block = 256;
  const int blocks = std::min(80 * 8, 65535); // cap gridDim.x

  CUDA_CHECK(cudaEventRecord(start));
  mc_kernel<<<blocks, block>>>(
      p.paths, p.steps, p.S0, p.K, p.r, p.sigma, p.T,
      (p.type == OptionType::EuropeanCall) ? 0 : 1,
      p.seed, d_sum, d_sum_sq);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  double h_sum = 0.0, h_sum_sq = 0.0;
  CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&h_sum_sq, d_sum_sq, sizeof(double), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_sum));
  CUDA_CHECK(cudaFree(d_sum_sq));

  const double n = static_cast<double>(p.paths);
  const double mean = h_sum / n;
  const double ex2 = h_sum_sq / n;
  const double var = std::max(ex2 - mean * mean, 0.0);
  const double std_error = std::sqrt(var / n);

  MCResult r{};
  r.price = mean;
  r.std_error = std_error;
  r.ms = static_cast<double>(ms);
  return r;
}

// ----------------------------- CPU baseline -----------------------------

static MCResult run_cpu(const Params& p) {
  const double dt = p.T / static_cast<double>(p.steps);
  const double drift = (p.r - 0.5 * p.sigma * p.sigma) * dt;
  const double vol = p.sigma * std::sqrt(dt);
  const double disc = std::exp(-p.r * p.T);

  std::mt19937_64 rng(p.seed);
  std::normal_distribution<double> nd(0.0, 1.0);

  auto t0 = std::chrono::high_resolution_clock::now();
  double sum = 0.0, sum_sq = 0.0;

  for (std::int64_t i = 0; i < p.paths; ++i) {
    double S = p.S0;
    double avgS = 0.0;
    for (int j = 0; j < p.steps; ++j) {
      const double z = nd(rng);
      S *= std::exp(drift + vol * z);
      if (p.type == OptionType::AsianArithmeticCall) avgS += S;
    }

    double payoff = 0.0;
    if (p.type == OptionType::EuropeanCall) {
      payoff = std::max(S - p.K, 0.0);
    } else {
      payoff = std::max((avgS / p.steps) - p.K, 0.0);
    }
    payoff *= disc;
    sum += payoff;
    sum_sq += payoff * payoff;
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  const double n = static_cast<double>(p.paths);
  const double mean = sum / n;
  const double ex2 = sum_sq / n;
  const double var = std::max(ex2 - mean * mean, 0.0);
  const double std_error = std::sqrt(var / n);

  MCResult r{};
  r.price = mean;
  r.std_error = std_error;
  r.ms = ms;
  return r;
}

// ----------------------------- main -----------------------------

int main(int argc, char** argv) {
  Params p{};
  bool run_cpu_flag = false;
  if (!parse_args(argc, argv, p, run_cpu_flag)) {
    usage(argv[0]);
    return 2;
  }

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Option: " << ((p.type == OptionType::EuropeanCall) ? "EuropeanCall" : "AsianArithmeticCall") << "\n";
  std::cout << "Params: S0=" << p.S0 << " K=" << p.K << " r=" << p.r
            << " sigma=" << p.sigma << " T=" << p.T
            << " steps=" << p.steps << " paths=" << p.paths << "\n";

  // GPU run (primary)
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count <= 0) {
    std::cerr << "No CUDA devices found.\n";
    return 1;
  }

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::cout << "GPU: " << prop.name << " (cc " << prop.major << "." << prop.minor << ")\n";

  MCResult g = run_gpu(p);
  std::cout << "[GPU] price=" << g.price << "  std_error=" << g.std_error
            << "  time_ms=" << g.ms << "\n";

  // Optional CPU baseline for sanity checks (slow for large paths)
  if (run_cpu_flag) {
    MCResult c = run_cpu(p);
    std::cout << "[CPU] price=" << c.price << "  std_error=" << c.std_error
              << "  time_ms=" << c.ms << "\n";
  }

  return 0;
}


