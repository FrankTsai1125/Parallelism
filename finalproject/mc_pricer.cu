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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
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
  double mu = 0.0;       // real-world drift (annualized). Used for path simulation if enabled.
  double sigma = 0.2;
  double T = 1.0;
  int steps = 252;
  std::int64_t paths = 5'000'000;
  std::int64_t paths_total = 0; // if set and running multi-task, split total paths across tasks
  OptionType type = OptionType::EuropeanCall;
  std::uint64_t seed = 1234;

  // Multi-asset (for basket). assets=1 keeps the original single-asset model.
  int assets = 1;
  double rho = 0.0; // equicorrelation off-diagonal correlation

  // GPU tuning knobs (to "make GPU busier" by increasing active threads/blocks).
  int block_size = 256;
  int blocks_per_sm = 8;
  int device = -1; // -1 => auto (use SLURM_LOCALID if present, else 0)

  // Optional: load S0/sigma from Yahoo CSV (downloaded by tools/download_2330_yahoo.py).
  std::string yahoo_csv;
  bool yahoo_use_adj = false; // use "Adj Close" instead of "Close" if available
  int trading_days = 252;     // for annualizing sigma
  bool use_yahoo_mu = true;   // for path simulation: prefer mu estimated from yahoo if available

  // Optional: write one-line result CSV.
  std::string out_csv;
  bool out_csv_append = false;

  // Optional: dump simulated price paths (future trajectory) to CSV.
  std::string dump_paths_csv;
  int dump_paths = 100; // number of paths to dump (for visualization)

  // Optional: output aggregated mean/std trajectory (one row per step). This is the recommended
  // output for CPU vs GPU vs 8GPU performance comparisons because it avoids massive I/O.
  std::string avg_path_csv;
};

static void usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0
      << " [--type european|asian|basket] [--S0 N] [--K N] [--r N] [--sigma N] [--T N]\n"
      << "           [--mu N] [--steps N] [--paths N] [--paths-total N] [--seed N] [--assets N] [--rho R]\n"
      << "           [--block-size N] [--blocks-per-sm N] [--device N] [--cpu]\n";
  std::cerr
      << "           [--yahoo-csv PATH] [--yahoo-adj] [--trading-days N]\n"
      << "           [--out-csv PATH] [--append-csv]\n"
      << "           [--dump-paths-csv PATH] [--dump-paths N] [--no-yahoo-mu]\n"
      << "           [--avg-path-csv PATH]\n";
}

struct ArgFlags {
  bool S0_set = false;
  bool sigma_set = false;
  bool mu_set = false;
};

static bool parse_args(int argc, char** argv, Params& p, bool& run_cpu, ArgFlags& flags) {
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
      flags.S0_set = true;
    } else if (std::strcmp(argv[i], "--K") == 0) {
      const char* v = next("--K"); if (!v) return false;
      p.K = std::stod(v);
    } else if (std::strcmp(argv[i], "--r") == 0) {
      const char* v = next("--r"); if (!v) return false;
      p.r = std::stod(v);
    } else if (std::strcmp(argv[i], "--mu") == 0) {
      const char* v = next("--mu"); if (!v) return false;
      p.mu = std::stod(v);
      flags.mu_set = true;
    } else if (std::strcmp(argv[i], "--sigma") == 0) {
      const char* v = next("--sigma"); if (!v) return false;
      p.sigma = std::stod(v);
      flags.sigma_set = true;
    } else if (std::strcmp(argv[i], "--T") == 0) {
      const char* v = next("--T"); if (!v) return false;
      p.T = std::stod(v);
    } else if (std::strcmp(argv[i], "--steps") == 0) {
      const char* v = next("--steps"); if (!v) return false;
      p.steps = std::stoi(v);
    } else if (std::strcmp(argv[i], "--paths") == 0) {
      const char* v = next("--paths"); if (!v) return false;
      p.paths = std::stoll(v);
    } else if (std::strcmp(argv[i], "--paths-total") == 0) {
      const char* v = next("--paths-total"); if (!v) return false;
      p.paths_total = std::stoll(v);
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
    } else if (std::strcmp(argv[i], "--yahoo-csv") == 0) {
      const char* v = next("--yahoo-csv"); if (!v) return false;
      p.yahoo_csv = v;
    } else if (std::strcmp(argv[i], "--yahoo-adj") == 0) {
      p.yahoo_use_adj = true;
    } else if (std::strcmp(argv[i], "--trading-days") == 0) {
      const char* v = next("--trading-days"); if (!v) return false;
      p.trading_days = std::stoi(v);
    } else if (std::strcmp(argv[i], "--out-csv") == 0) {
      const char* v = next("--out-csv"); if (!v) return false;
      p.out_csv = v;
    } else if (std::strcmp(argv[i], "--append-csv") == 0) {
      p.out_csv_append = true;
    } else if (std::strcmp(argv[i], "--dump-paths-csv") == 0) {
      const char* v = next("--dump-paths-csv"); if (!v) return false;
      p.dump_paths_csv = v;
    } else if (std::strcmp(argv[i], "--dump-paths") == 0) {
      const char* v = next("--dump-paths"); if (!v) return false;
      p.dump_paths = std::stoi(v);
    } else if (std::strcmp(argv[i], "--no-yahoo-mu") == 0) {
      p.use_yahoo_mu = false;
    } else if (std::strcmp(argv[i], "--avg-path-csv") == 0) {
      const char* v = next("--avg-path-csv"); if (!v) return false;
      p.avg_path_csv = v;
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
  if (p.paths_total < 0) {
    std::cerr << "Invalid --paths-total.\n";
    return false;
  }
  if (p.block_size <= 0 || p.blocks_per_sm <= 0) {
    std::cerr << "Invalid GPU launch parameters.\n";
    return false;
  }
  if (p.trading_days <= 0) {
    std::cerr << "Invalid --trading-days.\n";
    return false;
  }
  if (p.dump_paths < 1) {
    std::cerr << "Invalid --dump-paths.\n";
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

static bool file_nonempty(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f.good()) return false;
  f.seekg(0, std::ios::end);
  return f.tellg() > 0;
}

static std::string csv_escape(std::string s) {
  bool need_quotes = false;
  for (char c : s) {
    if (c == '"' || c == ',' || c == '\n' || c == '\r') { need_quotes = true; break; }
  }
  if (!need_quotes) return s;
  std::string out;
  out.reserve(s.size() + 2);
  out.push_back('"');
  for (char c : s) {
    if (c == '"') out += "\"\"";
    else out.push_back(c);
  }
  out.push_back('"');
  return out;
}

struct YahooStats {
  double S0 = 0.0;
  double sigma_daily = 0.0;
  double sigma_annual = 0.0;
  double mu_daily = 0.0;
  double mu_annual = 0.0;
  int n_prices = 0;
  int n_returns = 0;
  std::string col_used;
};

static YahooStats load_yahoo_csv_compute_S0_sigma(const std::string& path, bool use_adj, int trading_days) {
  std::ifstream in(path);
  if (!in.is_open()) {
    throw std::runtime_error("Failed to open yahoo csv: " + path);
  }

  std::string header;
  if (!std::getline(in, header)) {
    throw std::runtime_error("Empty yahoo csv: " + path);
  }

  auto split = [](const std::string& line) -> std::vector<std::string> {
    std::vector<std::string> out;
    std::string cur;
    std::istringstream ss(line);
    while (std::getline(ss, cur, ',')) out.push_back(cur);
    return out;
  };

  std::vector<std::string> cols = split(header);
  int idx_close = -1;
  int idx_adj = -1;
  for (int i = 0; i < static_cast<int>(cols.size()); ++i) {
    if (cols[i] == "Close") idx_close = i;
    if (cols[i] == "Adj Close") idx_adj = i;
  }
  int idx = (use_adj && idx_adj >= 0) ? idx_adj : idx_close;
  std::string col_used = (use_adj && idx_adj >= 0) ? "Adj Close" : "Close";
  if (idx < 0) {
    throw std::runtime_error("Yahoo csv missing required column Close/Adj Close: " + path);
  }

  std::vector<double> px;
  px.reserve(2048);

  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::vector<std::string> fields = split(line);
    if (idx >= static_cast<int>(fields.size())) continue;
    try {
      const std::string& s = fields[idx];
      if (s.empty()) continue;
      double v = std::stod(s);
      if (v > 0.0 && std::isfinite(v)) px.push_back(v);
    } catch (...) {
      continue;
    }
  }

  if (px.size() < 2) {
    throw std::runtime_error("Not enough price points in yahoo csv: " + path);
  }

  std::vector<double> r;
  r.reserve(px.size() - 1);
  for (std::size_t i = 1; i < px.size(); ++i) {
    r.push_back(std::log(px[i] / px[i - 1]));
  }

  // Sample stddev (ddof=1)
  double mean = 0.0;
  for (double x : r) mean += x;
  mean /= static_cast<double>(r.size());

  double ssq = 0.0;
  for (double x : r) {
    double d = x - mean;
    ssq += d * d;
  }
  double var = ssq / static_cast<double>(r.size() - 1);
  double sigma_daily = std::sqrt(std::max(var, 0.0));
  double sigma_annual = sigma_daily * std::sqrt(static_cast<double>(trading_days));
  double mu_daily = mean;
  double mu_annual = mu_daily * static_cast<double>(trading_days);

  YahooStats st{};
  st.S0 = px.back();
  st.sigma_daily = sigma_daily;
  st.sigma_annual = sigma_annual;
  st.mu_daily = mu_daily;
  st.mu_annual = mu_annual;
  st.n_prices = static_cast<int>(px.size());
  st.n_returns = static_cast<int>(r.size());
  st.col_used = col_used;
  return st;
}

static const char* option_type_str(OptionType t) {
  switch (t) {
    case OptionType::EuropeanCall: return "european";
    case OptionType::AsianArithmeticCall: return "asian";
    case OptionType::BasketEuropeanCall: return "basket";
  }
  return "unknown";
}

static std::string now_timestamp_utc() {
  using clock = std::chrono::system_clock;
  auto now = clock::now();
  std::time_t tt = clock::to_time_t(now);
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &tt);
#else
  gmtime_r(&tt, &tm);
#endif
  std::ostringstream oss;
  oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
  return oss.str();
}

struct MCResult {
  double price = 0.0;
  double std_error = 0.0;
  double ms = 0.0;
};

static void write_result_csv(
    const Params& p,
    const MCResult& g,
    const MCResult* c,
    const cudaDeviceProp& prop,
    int chosen_device,
    int proc_id,
    int local_id,
    const YahooStats* ys) {

  const bool append = p.out_csv_append;
  const bool need_header = !append || !file_nonempty(p.out_csv);

  std::ofstream out(p.out_csv, append ? (std::ios::out | std::ios::app) : (std::ios::out | std::ios::trunc));
  if (!out.is_open()) {
    throw std::runtime_error("Failed to open out csv: " + p.out_csv);
  }

  if (need_header) {
    out << "timestamp_utc,"
        << "yahoo_csv,yahoo_col,yahoo_S0,yahoo_sigma_annual,yahoo_mu_annual,yahoo_n_prices,yahoo_n_returns,"
        << "S0,sigma,mu,K,r,T,steps,paths,type,assets,rho,block_size,blocks_per_sm,"
        << "device,slurm_procid,slurm_localid,seed,"
        << "gpu_name,cc_major,cc_minor,"
        << "gpu_price,gpu_std_error,gpu_time_ms,"
        << "cpu_price,cpu_std_error,cpu_time_ms"
        << "\n";
  }

  out << csv_escape(now_timestamp_utc()) << ",";

  if (ys) {
    out << csv_escape(p.yahoo_csv) << ","
        << csv_escape(ys->col_used) << ","
        << ys->S0 << ","
        << ys->sigma_annual << ","
        << ys->mu_annual << ","
        << ys->n_prices << ","
        << ys->n_returns << ",";
  } else {
    out << "," << "," << "," << "," << "," << "," << ",";
  }

  out << p.S0 << "," << p.sigma << "," << p.mu << "," << p.K << "," << p.r << "," << p.T << ","
      << p.steps << "," << p.paths << "," << csv_escape(option_type_str(p.type)) << ","
      << p.assets << "," << p.rho << "," << p.block_size << "," << p.blocks_per_sm << ","
      << chosen_device << "," << proc_id << "," << local_id << "," << p.seed << ","
      << csv_escape(prop.name) << "," << prop.major << "," << prop.minor << ","
      << g.price << "," << g.std_error << "," << g.ms << ",";

  if (c) {
    out << c->price << "," << c->std_error << "," << c->ms;
  } else {
    out << "," << ",";
  }
  out << "\n";
}

// For --avg-path-csv mode (trajectory output), we still want per-run performance logs.
// This uses the same schema as write_result_csv, but writes "avg_path" as type and
// leaves option-price fields blank.
static void write_avgpath_bench_csv_gpu(
    const Params& p,
    std::int64_t paths_logged,
    double gpu_time_ms,
    const cudaDeviceProp& prop,
    int chosen_device,
    int proc_id,
    int local_id,
    const YahooStats* ys) {

  if (p.out_csv.empty()) return;
  const bool append = p.out_csv_append;
  const bool need_header = !append || !file_nonempty(p.out_csv);

  std::ofstream out(p.out_csv, append ? (std::ios::out | std::ios::app) : (std::ios::out | std::ios::trunc));
  if (!out.is_open()) {
    throw std::runtime_error("Failed to open out csv: " + p.out_csv);
  }

  if (need_header) {
    out << "timestamp_utc,"
        << "yahoo_csv,yahoo_col,yahoo_S0,yahoo_sigma_annual,yahoo_mu_annual,yahoo_n_prices,yahoo_n_returns,"
        << "S0,sigma,mu,K,r,T,steps,paths,type,assets,rho,block_size,blocks_per_sm,"
        << "device,slurm_procid,slurm_localid,seed,"
        << "gpu_name,cc_major,cc_minor,"
        << "gpu_price,gpu_std_error,gpu_time_ms,"
        << "cpu_price,cpu_std_error,cpu_time_ms"
        << "\n";
  }

  out << csv_escape(now_timestamp_utc()) << ",";

  if (ys) {
    out << csv_escape(p.yahoo_csv) << ","
        << csv_escape(ys->col_used) << ","
        << ys->S0 << ","
        << ys->sigma_annual << ","
        << ys->mu_annual << ","
        << ys->n_prices << ","
        << ys->n_returns << ",";
  } else {
    out << "," << "," << "," << "," << "," << "," << ",";
  }

  // Keep schema aligned with option-pricing CSV, but mark type as avg_path and omit option-price fields.
  out << p.S0 << "," << p.sigma << "," << p.mu << ","
      << "" << "," // K
      << p.r << "," << p.T << "," << p.steps << "," << paths_logged << ","
      << "avg_path" << "," // type
      << "" << ","         // assets
      << "" << ","         // rho
      << p.block_size << "," << "" << "," // block_size, blocks_per_sm (avg-path doesn't use blocks_per_sm)
      << chosen_device << "," << proc_id << "," << local_id << "," << p.seed << ","
      << csv_escape(prop.name) << "," << prop.major << "," << prop.minor << ","
      << "" << "," << "" << "," << gpu_time_ms << "," // gpu price/std/time
      << "" << "," << "" << "," << ""                // cpu price/std/time
      << "\n";
}

static void dump_paths_csv(
    const Params& p,
    const YahooStats* ys,
    bool have_yahoo,
    bool mu_overridden_by_cli) {

  // For "forecast-like" simulation we use mu (real-world drift) if provided, else
  // optionally use yahoo-estimated mu, else fall back to r.
  double drift_rate = p.r;
  if (mu_overridden_by_cli) {
    drift_rate = p.mu;
  } else if (have_yahoo && p.use_yahoo_mu) {
    drift_rate = ys->mu_annual;
  } else {
    drift_rate = p.r;
  }

  std::ofstream out(p.dump_paths_csv, std::ios::out | std::ios::trunc);
  if (!out.is_open()) {
    throw std::runtime_error("Failed to open dump csv: " + p.dump_paths_csv);
  }
  out << "path_id,step,t_years,price\n";

  // Simple CPU simulation for export (small N like 100 paths is fine).
  const double dt = p.T / static_cast<double>(p.steps);
  const double drift = (drift_rate - 0.5 * p.sigma * p.sigma) * dt;
  const double vol = p.sigma * std::sqrt(dt);

  std::mt19937_64 rng(p.seed);
  std::normal_distribution<double> nd(0.0, 1.0);

  for (int pid = 0; pid < p.dump_paths; ++pid) {
    double S = p.S0;
    out << pid << "," << 0 << "," << 0.0 << "," << S << "\n";
    for (int t = 1; t <= p.steps; ++t) {
      const double z = nd(rng);
      S *= std::exp(drift + vol * z);
      out << pid << "," << t << "," << (static_cast<double>(t) * dt) << "," << S << "\n";
    }
  }
}

// ----------------------------- GPU mean/std trajectory (single-asset) -----------------------------

__global__ void meanpath_kernel(
    std::int64_t paths,
    int steps,
    double S0,
    double drift_rate, // annualized drift (mu or r)
    double sigma,
    double T,
    std::uint64_t seed,
    double* sumS,      // length steps+1
    double* sumSqS) {  // length steps+1

  const int tid = threadIdx.x;
  const std::int64_t gid = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + tid;
  if (gid >= paths) return;

  extern __shared__ double sh[];
  double* sh_sum = sh;
  double* sh_sumsq = sh + blockDim.x;

  const double dt = T / static_cast<double>(steps);
  const double drift = (drift_rate - 0.5 * sigma * sigma) * dt;
  const double vol = sigma * std::sqrt(dt);

  curandStatePhilox4_32_10_t st;
  curand_init(static_cast<unsigned long long>(seed),
              static_cast<unsigned long long>(gid),
              0ULL, &st);

  double S = S0;

  // step 0
  sh_sum[tid] = S;
  sh_sumsq[tid] = S * S;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      sh_sum[tid] += sh_sum[tid + offset];
      sh_sumsq[tid] += sh_sumsq[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd(&sumS[0], sh_sum[0]);
    atomicAdd(&sumSqS[0], sh_sumsq[0]);
  }

  for (int t = 1; t <= steps; ++t) {
    const double z = static_cast<double>(curand_normal(&st));
    S *= std::exp(drift + vol * z);

    sh_sum[tid] = S;
    sh_sumsq[tid] = S * S;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
      if (tid < offset) {
        sh_sum[tid] += sh_sum[tid + offset];
        sh_sumsq[tid] += sh_sumsq[tid + offset];
      }
      __syncthreads();
    }
    if (tid == 0) {
      atomicAdd(&sumS[t], sh_sum[0]);
      atomicAdd(&sumSqS[t], sh_sumsq[0]);
    }
  }
}

struct MeanPathResult {
  std::vector<double> mean; // length steps+1
  std::vector<double> std;  // length steps+1
  double ms = 0.0;
};

static MeanPathResult run_gpu_meanpath_single_asset(const Params& p, double drift_rate) {
  if (p.paths <= 0) {
    throw std::runtime_error("paths must be > 0 for avg-path-csv");
  }
  if (p.steps <= 0) {
    throw std::runtime_error("steps must be > 0 for avg-path-csv");
  }

  const int block = p.block_size;
  const std::int64_t blocks_needed = (p.paths + block - 1) / block;
  if (blocks_needed > 65535) {
    throw std::runtime_error("Too many blocks for avg-path-csv with current block_size; reduce paths or increase block_size.");
  }
  const int blocks = static_cast<int>(blocks_needed);

  const std::size_t n = static_cast<std::size_t>(p.steps) + 1;
  double* d_sum = nullptr;
  double* d_sumsq = nullptr;
  CUDA_CHECK(cudaMalloc(&d_sum, sizeof(double) * n));
  CUDA_CHECK(cudaMalloc(&d_sumsq, sizeof(double) * n));
  CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(double) * n));
  CUDA_CHECK(cudaMemset(d_sumsq, 0, sizeof(double) * n));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  const std::size_t shmem = static_cast<std::size_t>(2) * static_cast<std::size_t>(block) * sizeof(double);

  CUDA_CHECK(cudaEventRecord(start));
  meanpath_kernel<<<blocks, block, shmem>>>(
      p.paths, p.steps, p.S0, drift_rate, p.sigma, p.T,
      p.seed, d_sum, d_sumsq);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  std::vector<double> h_sum(n, 0.0);
  std::vector<double> h_sumsq(n, 0.0);
  CUDA_CHECK(cudaMemcpy(h_sum.data(), d_sum, sizeof(double) * n, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_sumsq.data(), d_sumsq, sizeof(double) * n, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_sum));
  CUDA_CHECK(cudaFree(d_sumsq));

  MeanPathResult r{};
  r.mean.resize(n);
  r.std.resize(n);
  const double dn = static_cast<double>(p.paths);
  for (std::size_t i = 0; i < n; ++i) {
    const double m = h_sum[i] / dn;
    const double ex2 = h_sumsq[i] / dn;
    const double var = std::max(ex2 - m * m, 0.0);
    r.mean[i] = m;
    r.std[i] = std::sqrt(var);
  }
  r.ms = static_cast<double>(ms);
  return r;
}

static void write_avg_path_csv(const std::string& path, int steps, double T,
                               const std::vector<double>& mean,
                               const std::vector<double>& stdev) {
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out.is_open()) throw std::runtime_error("Failed to open avg-path csv: " + path);
  out << "step,t_years,mean,std\n";
  const double dt = T / static_cast<double>(steps);
  for (int s = 0; s <= steps; ++s) {
    out << s << "," << (static_cast<double>(s) * dt) << "," << mean[static_cast<std::size_t>(s)]
        << "," << stdev[static_cast<std::size_t>(s)] << "\n";
  }
}

static void write_avg_partial_bin(const std::string& path, int steps, std::int64_t local_paths,
                                  const std::vector<double>& sum,
                                  const std::vector<double>& sumsq) {
  std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
  if (!out.is_open()) throw std::runtime_error("Failed to open partial file: " + path);
  out.write(reinterpret_cast<const char*>(&steps), sizeof(int));
  out.write(reinterpret_cast<const char*>(&local_paths), sizeof(std::int64_t));
  const std::size_t n = static_cast<std::size_t>(steps) + 1;
  out.write(reinterpret_cast<const char*>(sum.data()), sizeof(double) * n);
  out.write(reinterpret_cast<const char*>(sumsq.data()), sizeof(double) * n);
}

static bool read_avg_partial_bin(const std::string& path, int& steps, std::int64_t& local_paths,
                                 std::vector<double>& sum, std::vector<double>& sumsq) {
  std::ifstream in(path, std::ios::in | std::ios::binary);
  if (!in.is_open()) return false;
  in.read(reinterpret_cast<char*>(&steps), sizeof(int));
  in.read(reinterpret_cast<char*>(&local_paths), sizeof(std::int64_t));
  const std::size_t n = static_cast<std::size_t>(steps) + 1;
  sum.resize(n);
  sumsq.resize(n);
  in.read(reinterpret_cast<char*>(sum.data()), sizeof(double) * n);
  in.read(reinterpret_cast<char*>(sumsq.data()), sizeof(double) * n);
  return in.good();
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
  ArgFlags flags{};
  if (!parse_args(argc, argv, p, run_cpu_flag, flags)) {
    usage(argv[0]);
    return 2;
  }

  YahooStats ys{};
  bool have_yahoo = false;
  if (!p.yahoo_csv.empty()) {
    try {
      ys = load_yahoo_csv_compute_S0_sigma(p.yahoo_csv, p.yahoo_use_adj, p.trading_days);
      have_yahoo = true;
      if (!flags.S0_set) p.S0 = ys.S0;
      if (!flags.sigma_set) p.sigma = ys.sigma_annual;
      if (!flags.mu_set) p.mu = ys.mu_annual;
    } catch (const std::exception& e) {
      std::cerr << "Yahoo CSV error: " << e.what() << "\n";
      return 2;
    }
  }

  // If launched via Slurm with multiple tasks, bind each task to its local GPU.
  // Taiwania2 nodes have 8x V100, so you can use --gpus-per-node 8 and --ntasks-per-node 8.
  const int local_id = env_int("SLURM_LOCALID", -1);
  const int proc_id = env_int("SLURM_PROCID", 0);
  const int ntasks = env_int("SLURM_NTASKS", 1);

  std::cout << std::fixed << std::setprecision(6);
  if (have_yahoo) {
    std::cout << "YahooCSV: path=" << p.yahoo_csv
              << " col=" << ys.col_used
              << " n_prices=" << ys.n_prices
              << " n_returns=" << ys.n_returns
              << " sigma_daily=" << ys.sigma_daily
              << " sigma_annual=" << ys.sigma_annual
              << " mu_annual=" << ys.mu_annual
              << " (trading_days=" << p.trading_days << ")\n";
    std::cout << "Using: S0=" << p.S0 << " sigma=" << p.sigma
              << (flags.S0_set ? " (S0 overridden by --S0)" : "")
              << (flags.sigma_set ? " (sigma overridden by --sigma)" : "")
              << " mu=" << p.mu
              << (flags.mu_set ? " (mu overridden by --mu)" : "")
              << "\n";
  }

  // Avg-path CSV mode (mean/std trajectory). Recommended for CPU vs GPU vs 8GPU scaling comparisons.
  if (!p.avg_path_csv.empty()) {
    if (p.assets != 1) {
      std::cerr << "--avg-path-csv is currently implemented for assets=1 only.\n";
      return 2;
    }

    // Determine local paths (for multi-task runs) and total paths for normalization.
    std::int64_t local_paths = p.paths;
    if (ntasks > 1 && p.paths_total > 0) {
      const std::int64_t base = p.paths_total / ntasks;
      const std::int64_t rem = p.paths_total % ntasks;
      local_paths = base + ((proc_id < rem) ? 1 : 0);
    }
    if (local_paths <= 0) {
      std::cerr << "Computed local_paths <= 0. Check --paths-total / ntasks.\n";
      return 2;
    }

    // Choose drift rate for simulation.
    const bool mu_overridden_by_cli = flags.mu_set;
    double drift_rate = p.r;
    if (mu_overridden_by_cli) drift_rate = p.mu;
    else if (have_yahoo && p.use_yahoo_mu) drift_rate = ys.mu_annual;
    else drift_rate = p.r;

    // Run GPU for local_paths.
    Params lp = p;
    lp.paths = local_paths;

    MeanPathResult mp = run_gpu_meanpath_single_asset(lp, drift_rate);

    // Convert mean/std back to sums for reduction (so proc0 can combine).
    const std::size_t n = static_cast<std::size_t>(p.steps) + 1;
    std::vector<double> sum(n, 0.0), sumsq(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      sum[i] = mp.mean[i] * static_cast<double>(local_paths);
      // mp.std is population std; reconstruct E[S^2] = Var + mean^2
      const double ex2 = (mp.std[i] * mp.std[i]) + (mp.mean[i] * mp.mean[i]);
      sumsq[i] = ex2 * static_cast<double>(local_paths);
    }

    // Multi-task reduction: each task writes a partial; proc0 merges.
    if (ntasks > 1) {
      const std::string partial = p.avg_path_csv + ".partial." + std::to_string(proc_id) + ".bin";
      write_avg_partial_bin(partial, p.steps, local_paths, sum, sumsq);

      if (proc_id == 0) {
        // wait for other partial files
        for (int pid = 0; pid < ntasks; ++pid) {
          const std::string f = p.avg_path_csv + ".partial." + std::to_string(pid) + ".bin";
          int tries = 0;
          while (!file_nonempty(f) && tries < 600) { // up to ~10 min
            std::this_thread::sleep_for(std::chrono::seconds(1));
            ++tries;
          }
          if (!file_nonempty(f)) {
            std::cerr << "Timeout waiting for partial: " << f << "\n";
            return 2;
          }
        }

        int steps_chk = -1;
        std::int64_t total_paths_acc = 0;
        std::vector<double> sum_all(n, 0.0), sumsq_all(n, 0.0);
        for (int pid = 0; pid < ntasks; ++pid) {
          const std::string f = p.avg_path_csv + ".partial." + std::to_string(pid) + ".bin";
          int st = 0;
          std::int64_t lp_count = 0;
          std::vector<double> s, ss;
          if (!read_avg_partial_bin(f, st, lp_count, s, ss)) {
            std::cerr << "Failed to read partial: " << f << "\n";
            return 2;
          }
          if (steps_chk < 0) steps_chk = st;
          if (st != steps_chk) {
            std::cerr << "Partial steps mismatch in " << f << "\n";
            return 2;
          }
          total_paths_acc += lp_count;
          for (std::size_t i = 0; i < n; ++i) {
            sum_all[i] += s[i];
            sumsq_all[i] += ss[i];
          }
          std::remove(f.c_str());
        }

        // Normalize to mean/std and write final CSV.
        std::vector<double> mean(n, 0.0), stdev(n, 0.0);
        const double dn = static_cast<double>(total_paths_acc);
        for (std::size_t i = 0; i < n; ++i) {
          const double m = sum_all[i] / dn;
          const double ex2 = sumsq_all[i] / dn;
          const double var = std::max(ex2 - m * m, 0.0);
          mean[i] = m;
          stdev[i] = std::sqrt(var);
        }
        write_avg_path_csv(p.avg_path_csv, p.steps, p.T, mean, stdev);
        std::cout << "Wrote avg-path CSV (reduced): " << p.avg_path_csv
                  << " total_paths=" << total_paths_acc
                  << " steps=" << p.steps
                  << " gpu_time_ms_local=" << mp.ms << "\n";

        // Append performance row to --out-csv (proc0 only).
        if (!p.out_csv.empty()) {
          int dev = 0;
          cudaDeviceProp prop{};
          CUDA_CHECK(cudaGetDevice(&dev));
          CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
          try {
            write_avgpath_bench_csv_gpu(p, total_paths_acc, mp.ms, prop, dev, proc_id, local_id, have_yahoo ? &ys : nullptr);
            std::cout << "Appended bench CSV: " << p.out_csv << "\n";
          } catch (const std::exception& e) {
            std::cerr << "CSV write error: " << e.what() << "\n";
            return 2;
          }
        }
      }
    } else {
      write_avg_path_csv(p.avg_path_csv, p.steps, p.T, mp.mean, mp.std);
      std::cout << "Wrote avg-path CSV: " << p.avg_path_csv
                << " paths=" << local_paths
                << " steps=" << p.steps
                << " gpu_time_ms=" << mp.ms << "\n";

      // Append performance row to --out-csv.
      if (!p.out_csv.empty()) {
        int dev = 0;
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        try {
          write_avgpath_bench_csv_gpu(p, local_paths, mp.ms, prop, dev, proc_id, local_id, have_yahoo ? &ys : nullptr);
          std::cout << "Appended bench CSV: " << p.out_csv << "\n";
        } catch (const std::exception& e) {
          std::cerr << "CSV write error: " << e.what() << "\n";
          return 2;
        }
      }
    }

    return 0;
  }

  // If requested, dump simulated future paths to CSV (independent from option pricing).
  if (!p.dump_paths_csv.empty()) {
    try {
      dump_paths_csv(p, &ys, have_yahoo, flags.mu_set);
      std::cout << "Wrote paths CSV: " << p.dump_paths_csv
                << " (paths=" << p.dump_paths << ", steps=" << p.steps
                << ", T=" << p.T << ")\n";
    } catch (const std::exception& e) {
      std::cerr << "Path dump error: " << e.what() << "\n";
      return 2;
    }
  }
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
  MCResult c{};
  bool have_cpu = false;
  if (run_cpu_flag) {
    c = run_cpu(p);
    have_cpu = true;
    std::cout << "[CPU] price=" << c.price << "  std_error=" << c.std_error
              << "  time_ms=" << c.ms << "\n";
  }

  if (!p.out_csv.empty()) {
    try {
      // In multi-task runs (e.g., 2GPU/8GPU), multiple tasks writing the same CSV would race.
      // Only let proc0 write the summary row.
      if (ntasks > 1 && proc_id != 0) {
        std::cout << "Skipping result CSV write on proc " << proc_id
                  << " (ntasks=" << ntasks << ", only proc0 writes): " << p.out_csv << "\n";
      } else {
        write_result_csv(p, g, have_cpu ? &c : nullptr, prop, chosen_device, proc_id, local_id, have_yahoo ? &ys : nullptr);
        std::cout << "Wrote result CSV: " << p.out_csv << (p.out_csv_append ? " (append)\n" : "\n");
      }
    } catch (const std::exception& e) {
      std::cerr << "CSV write error: " << e.what() << "\n";
      return 2;
    }
  }

  return 0;
}


