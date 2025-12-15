// Monte Carlo option pricing (European / Asian / Basket) on NVIDIA Tesla V100 (sm_70).
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
//   ./mc_pricer --paths 2000000 --steps 252 --type basket --assets 16 --rho 0.3
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
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
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

enum class OptionType { EuropeanCall, AsianArithmeticCall, BasketEuropeanCall };

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

  // Multi-asset (for basket). assets=1 keeps the original single-asset model.
  int assets = 1;
  double rho = 0.0; // equicorrelation off-diagonal correlation

  // GPU tuning knobs (to "make GPU busier" by increasing active threads/blocks).
  int block_size = 256;
  int blocks_per_sm = 8;
  int device = -1; // -1 => auto (use SLURM_LOCALID if present, else 0)
};

static void usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0
      << " [--type european|asian|basket] [--S0 N] [--K N] [--r N] [--sigma N] [--T N]\n"
      << "           [--steps N] [--paths N] [--seed N] [--assets N] [--rho R]\n"
      << "           [--block-size N] [--blocks-per-sm N] [--device N] [--cpu]\n";
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
      else if (std::strcmp(v, "basket") == 0) p.type = OptionType::BasketEuropeanCall;
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
    } else if (std::strcmp(argv[i], "--assets") == 0) {
      const char* v = next("--assets"); if (!v) return false;
      p.assets = std::stoi(v);
    } else if (std::strcmp(argv[i], "--rho") == 0) {
      const char* v = next("--rho"); if (!v) return false;
      p.rho = std::stod(v);
    } else if (std::strcmp(argv[i], "--block-size") == 0) {
      const char* v = next("--block-size"); if (!v) return false;
      p.block_size = std::stoi(v);
    } else if (std::strcmp(argv[i], "--blocks-per-sm") == 0) {
      const char* v = next("--blocks-per-sm"); if (!v) return false;
      p.blocks_per_sm = std::stoi(v);
    } else if (std::strcmp(argv[i], "--device") == 0) {
      const char* v = next("--device"); if (!v) return false;
      p.device = std::stoi(v);
    } else if (std::strcmp(argv[i], "--cpu") == 0) {
      run_cpu = true;
    } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      return false;
    } else {
      std::cerr << "Unknown arg: " << argv[i] << "\n";
      return false;
    }
  }
  if (p.steps <= 0 || p.paths <= 0 || p.T <= 0.0 || p.sigma < 0.0 || p.assets <= 0) {
    std::cerr << "Invalid parameters.\n";
    return false;
  }
  if (p.block_size <= 0 || p.blocks_per_sm <= 0) {
    std::cerr << "Invalid GPU launch parameters.\n";
    return false;
  }
  if (p.type == OptionType::AsianArithmeticCall && p.assets != 1) {
    std::cerr << "Asian option is currently implemented for assets=1 only.\n";
    return false;
  }
  if (p.type == OptionType::BasketEuropeanCall && p.assets < 2) {
    // allow assets=1 but it's identical to european; warn by allowing it.
  }
  return true;
}

static int env_int(const char* name, int fallback) {
  const char* v = std::getenv(name);
  if (!v || !*v) return fallback;
  try {
    return std::stoi(v);
  } catch (...) {
    return fallback;
  }
}

static std::vector<double> make_equicorr_matrix(int n, double rho) {
  std::vector<double> A(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      A[static_cast<std::size_t>(i) * n + j] = (i == j) ? 1.0 : rho;
    }
  }
  return A;
}

// Simple CPU Cholesky (lower-triangular) for SPD matrices.
static std::vector<double> cholesky_lower(const std::vector<double>& A, int n) {
  std::vector<double> L(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      double sum = A[static_cast<std::size_t>(i) * n + j];
      for (int k = 0; k < j; ++k) {
        sum -= L[static_cast<std::size_t>(i) * n + k] * L[static_cast<std::size_t>(j) * n + k];
      }
      if (i == j) {
        if (sum <= 0.0) {
          throw std::runtime_error("Cholesky failed: matrix not SPD (try different rho/assets).");
        }
        L[static_cast<std::size_t>(i) * n + j] = std::sqrt(sum);
      } else {
        L[static_cast<std::size_t>(i) * n + j] = sum / L[static_cast<std::size_t>(j) * n + j];
      }
    }
  }
  return L;
}

// ----------------------------- GPU implementation -----------------------------

static constexpr int MAX_ASSETS = 32;

__global__ void mc_kernel(
    std::int64_t paths,
    int steps,
    double S0, double K, double r, double sigma, double T,
    int opt_type, // 0: EuropeanCall, 1: AsianArithmeticCall, 2: BasketEuropeanCall
    int assets,
    const double* __restrict__ chol_L, // assets x assets, lower-triangular (row-major); null if assets==1
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
    if (assets <= 1) {
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
      if (opt_type == 0 || opt_type == 2) {
        payoff = fmax(S - K, 0.0);
      } else {
        const double meanS = avgS / static_cast<double>(steps);
        payoff = fmax(meanS - K, 0.0);
      }
      payoff *= disc;

      local_sum += payoff;
      local_sum_sq += payoff * payoff;
      continue;
    }

    // Multi-asset basket (terminal arithmetic mean across assets).
    double Svec[MAX_ASSETS];
    for (int a = 0; a < assets; ++a) Svec[a] = S0;

    // Temporary vectors.
    double zvec[MAX_ASSETS];
    double yvec[MAX_ASSETS];

    for (int t = 0; t < steps; ++t) {
      // Generate independent normals z.
      int a = 0;
      for (; a + 4 <= assets; a += 4) {
        float4 z = curand_normal4(&st);
        zvec[a + 0] = static_cast<double>(z.x);
        zvec[a + 1] = static_cast<double>(z.y);
        zvec[a + 2] = static_cast<double>(z.z);
        zvec[a + 3] = static_cast<double>(z.w);
      }
      for (; a < assets; ++a) {
        zvec[a] = static_cast<double>(curand_normal(&st));
      }

      // Correlate: y = L * z (L is lower triangular).
      for (int i = 0; i < assets; ++i) {
        double acc = 0.0;
        const int row = i * assets;
        for (int k = 0; k <= i; ++k) {
          acc += chol_L[row + k] * zvec[k];
        }
        yvec[i] = acc;
      }

      // Update each asset.
      for (int i = 0; i < assets; ++i) {
        Svec[i] *= std::exp(drift + vol * yvec[i]);
      }
    }

    double basket = 0.0;
    for (int i = 0; i < assets; ++i) basket += Svec[i];
    basket /= static_cast<double>(assets);

    double payoff = fmax(basket - K, 0.0) * disc;

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

  // Optional multi-asset Cholesky matrix on device.
  double* d_L = nullptr;
  std::vector<double> h_L;
  if (p.assets > 1) {
    if (p.assets > MAX_ASSETS) {
      std::cerr << "assets too large (max " << MAX_ASSETS << ")\n";
      std::exit(1);
    }
    // Equicorrelation matrix is SPD only if rho in (-1/(n-1), 1).
    const double lower = -1.0 / static_cast<double>(p.assets - 1);
    if (!(p.rho > lower && p.rho < 1.0)) {
      std::cerr << "Invalid rho for equicorrelation SPD: require rho in (" << lower << ", 1)\n";
      std::exit(1);
    }
    try {
      auto A = make_equicorr_matrix(p.assets, p.rho);
      h_L = cholesky_lower(A, p.assets);
    } catch (const std::exception& e) {
      std::cerr << "Cholesky error: " << e.what() << "\n";
      std::exit(1);
    }
    CUDA_CHECK(cudaMalloc(&d_L, sizeof(double) * static_cast<std::size_t>(p.assets) * static_cast<std::size_t>(p.assets)));
    CUDA_CHECK(cudaMemcpy(d_L, h_L.data(),
                          sizeof(double) * static_cast<std::size_t>(p.assets) * static_cast<std::size_t>(p.assets),
                          cudaMemcpyHostToDevice));
  }

  const int block = p.block_size;
  const int blocks = std::min(prop.multiProcessorCount * p.blocks_per_sm, 65535); // cap gridDim.x

  CUDA_CHECK(cudaEventRecord(start));
  mc_kernel<<<blocks, block>>>(
      p.paths, p.steps, p.S0, p.K, p.r, p.sigma, p.T,
      (p.type == OptionType::EuropeanCall) ? 0 : (p.type == OptionType::AsianArithmeticCall ? 1 : 2),
      p.assets,
      d_L,
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
  if (d_L) CUDA_CHECK(cudaFree(d_L));

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

  // If launched via Slurm with multiple tasks, bind each task to its local GPU.
  // Taiwania2 nodes have 8x V100, so you can use --gpus-per-node 8 and --ntasks-per-node 8.
  const int local_id = env_int("SLURM_LOCALID", -1);
  const int proc_id = env_int("SLURM_PROCID", 0);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Option: "
            << ((p.type == OptionType::EuropeanCall)
                    ? "EuropeanCall"
                    : (p.type == OptionType::AsianArithmeticCall ? "AsianArithmeticCall" : "BasketEuropeanCall"))
            << "\n";
  std::cout << "Params: S0=" << p.S0 << " K=" << p.K << " r=" << p.r
            << " sigma=" << p.sigma << " T=" << p.T
            << " steps=" << p.steps << " paths=" << p.paths
            << " assets=" << p.assets << " rho=" << p.rho
            << " block_size=" << p.block_size << " blocks_per_sm=" << p.blocks_per_sm
            << "\n";

  // GPU run (primary)
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count <= 0) {
    std::cerr << "No CUDA devices found.\n";
    return 1;
  }

  int chosen_device = 0;
  if (p.device >= 0) {
    chosen_device = p.device;
  } else if (local_id >= 0) {
    chosen_device = local_id;
  }
  chosen_device = ((chosen_device % device_count) + device_count) % device_count;
  CUDA_CHECK(cudaSetDevice(chosen_device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, chosen_device));
  std::cout << "GPU: " << prop.name << " (cc " << prop.major << "." << prop.minor << ")"
            << " device=" << chosen_device;
  if (local_id >= 0) std::cout << " SLURM_LOCALID=" << local_id;
  if (proc_id >= 0) std::cout << " SLURM_PROCID=" << proc_id;
  std::cout << "\n";

  // Make seeds different across Slurm tasks (useful for multi-GPU runs without identical RNG streams).
  p.seed = p.seed ^ (0x9E3779B97F4A7C15ULL * static_cast<std::uint64_t>(proc_id + 1));

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


