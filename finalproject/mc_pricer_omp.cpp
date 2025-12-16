// CPU Monte Carlo path simulation (mean/std trajectory) using OpenMP.
// Designed to mirror the "avg-path-csv" output of mc_pricer.cu for fair CPU vs GPU comparisons.
//
// Build (Linux):
//   g++ -O3 -std=c++17 -fopenmp mc_pricer_omp.cpp -o mc_pricer_omp
//
// Example:
//   ./mc_pricer_omp --yahoo-csv data/2330_TW.csv --T 1 --steps 1000 --paths 10000000 --avg-path-csv avg_2330_cpu.csv

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <omp.h>

struct Params {
  double S0 = 100.0;
  double r = 0.05;
  double mu = 0.0;       // annualized drift (real-world)
  double sigma = 0.2;
  double T = 1.0;
  int steps = 252;
  std::int64_t paths = 5'000'000;
  std::int64_t paths_total = 0; // accepted for symmetry; on CPU we just use --paths
  std::uint64_t seed = 1234;

  std::string yahoo_csv;
  bool yahoo_use_adj = false;
  int trading_days = 252;
  bool use_yahoo_mu = true;

  std::string avg_path_csv;

  std::string out_csv;
  bool out_csv_append = false;

  int omp_threads = 0; // 0 => OMP default
};

struct ArgFlags {
  bool S0_set = false;
  bool sigma_set = false;
  bool mu_set = false;
};

static void usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0
      << " [--S0 N] [--r N] [--mu N] [--sigma N] [--T N]\n"
      << "           [--steps N] [--paths N] [--paths-total N] [--seed N]\n"
      << "           [--yahoo-csv PATH] [--yahoo-adj] [--trading-days N] [--no-yahoo-mu]\n"
      << "           [--avg-path-csv PATH]\n"
      << "           [--out-csv PATH] [--append-csv]\n"
      << "           [--omp-threads N]\n";
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

static int env_int(const char* name, int fallback) {
  const char* v = std::getenv(name);
  if (!v || !*v) return fallback;
  try {
    return std::stoi(v);
  } catch (...) {
    return fallback;
  }
}

struct YahooStats {
  double S0 = 0.0;
  double sigma_annual = 0.0;
  double mu_annual = 0.0;
  int n_prices = 0;
  int n_returns = 0;
  std::string col_used;
};

static YahooStats load_yahoo_csv_compute_stats(const std::string& path, bool use_adj, int trading_days) {
  std::ifstream in(path);
  if (!in.is_open()) throw std::runtime_error("Failed to open yahoo csv: " + path);

  std::string header;
  if (!std::getline(in, header)) throw std::runtime_error("Empty yahoo csv: " + path);

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
  const int idx = (use_adj && idx_adj >= 0) ? idx_adj : idx_close;
  const std::string col_used = (use_adj && idx_adj >= 0) ? "Adj Close" : "Close";
  if (idx < 0) throw std::runtime_error("Yahoo csv missing Close/Adj Close: " + path);

  std::vector<double> px;
  px.reserve(2048);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    auto fields = split(line);
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
  if (px.size() < 2) throw std::runtime_error("Not enough price points in yahoo csv: " + path);

  std::vector<double> r;
  r.reserve(px.size() - 1);
  for (std::size_t i = 1; i < px.size(); ++i) r.push_back(std::log(px[i] / px[i - 1]));

  double mean = 0.0;
  for (double x : r) mean += x;
  mean /= static_cast<double>(r.size());

  double ssq = 0.0;
  for (double x : r) {
    const double d = x - mean;
    ssq += d * d;
  }
  const double var = ssq / static_cast<double>(r.size() - 1);
  const double sigma_daily = std::sqrt(std::max(var, 0.0));

  YahooStats st{};
  st.S0 = px.back();
  st.sigma_annual = sigma_daily * std::sqrt(static_cast<double>(trading_days));
  st.mu_annual = mean * static_cast<double>(trading_days);
  st.n_prices = static_cast<int>(px.size());
  st.n_returns = static_cast<int>(r.size());
  st.col_used = col_used;
  return st;
}

static bool parse_args(int argc, char** argv, Params& p, ArgFlags& flags) {
  for (int i = 1; i < argc; ++i) {
    auto next = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << "\n";
        return nullptr;
      }
      return argv[++i];
    };

    if (std::strcmp(argv[i], "--S0") == 0) {
      const char* v = next("--S0"); if (!v) return false;
      p.S0 = std::stod(v);
      flags.S0_set = true;
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
    } else if (std::strcmp(argv[i], "--yahoo-csv") == 0) {
      const char* v = next("--yahoo-csv"); if (!v) return false;
      p.yahoo_csv = v;
    } else if (std::strcmp(argv[i], "--yahoo-adj") == 0) {
      p.yahoo_use_adj = true;
    } else if (std::strcmp(argv[i], "--trading-days") == 0) {
      const char* v = next("--trading-days"); if (!v) return false;
      p.trading_days = std::stoi(v);
    } else if (std::strcmp(argv[i], "--no-yahoo-mu") == 0) {
      p.use_yahoo_mu = false;
    } else if (std::strcmp(argv[i], "--avg-path-csv") == 0) {
      const char* v = next("--avg-path-csv"); if (!v) return false;
      p.avg_path_csv = v;
    } else if (std::strcmp(argv[i], "--out-csv") == 0) {
      const char* v = next("--out-csv"); if (!v) return false;
      p.out_csv = v;
    } else if (std::strcmp(argv[i], "--append-csv") == 0) {
      p.out_csv_append = true;
    } else if (std::strcmp(argv[i], "--omp-threads") == 0) {
      const char* v = next("--omp-threads"); if (!v) return false;
      p.omp_threads = std::stoi(v);
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
  if (p.trading_days <= 0) {
    std::cerr << "Invalid --trading-days.\n";
    return false;
  }
  return true;
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

static void write_bench_csv(const Params& p,
                            const YahooStats* ys,
                            bool have_yahoo,
                            double cpu_time_ms) {
  if (p.out_csv.empty()) return;
  const bool append = p.out_csv_append;
  const bool need_header = !append || !file_nonempty(p.out_csv);

  std::ofstream out(p.out_csv, append ? (std::ios::out | std::ios::app) : (std::ios::out | std::ios::trunc));
  if (!out.is_open()) throw std::runtime_error("Failed to open out csv: " + p.out_csv);

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

  const int proc_id = env_int("SLURM_PROCID", 0);
  const int local_id = env_int("SLURM_LOCALID", -1);
  out << csv_escape(now_timestamp_utc()) << ",";

  if (have_yahoo) {
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

  // Keep columns aligned with GPU CSV even though CPU doesn't use these fields.
  out << p.S0 << "," << p.sigma << "," << p.mu << ","
      << "" << "," // K
      << p.r << "," << p.T << "," << p.steps << "," << p.paths << ","
      << "avg_path" << "," // type
      << "" << ","         // assets
      << "" << ","         // rho
      << "" << ","         // block_size
      << "" << ","         // blocks_per_sm
      << "" << ","         // device
      << proc_id << "," << local_id << "," << p.seed << ","
      << "" << "," << "" << "," << "" << "," // gpu_name/cc
      << "" << "," << "" << "," << "" << "," // gpu price/std/time
      << "" << "," << "" << "," << cpu_time_ms
      << "\n";
}

int main(int argc, char** argv) {
  Params p{};
  ArgFlags flags{};
  if (!parse_args(argc, argv, p, flags)) {
    usage(argv[0]);
    return 2;
  }

  YahooStats ys{};
  bool have_yahoo = false;
  if (!p.yahoo_csv.empty()) {
    ys = load_yahoo_csv_compute_stats(p.yahoo_csv, p.yahoo_use_adj, p.trading_days);
    have_yahoo = true;
    if (!flags.S0_set) p.S0 = ys.S0;
    if (!flags.sigma_set) p.sigma = ys.sigma_annual;
    if (!flags.mu_set) p.mu = ys.mu_annual;
  }

  if (p.omp_threads > 0) omp_set_num_threads(p.omp_threads);
  const int threads = omp_get_max_threads();

  if (p.avg_path_csv.empty()) {
    std::cerr << "Missing --avg-path-csv (CPU OpenMP version only supports avg-path output).\n";
    return 2;
  }

  // Choose drift rate for simulation.
  double drift_rate = (have_yahoo && p.use_yahoo_mu && !flags.mu_set) ? ys.mu_annual : (flags.mu_set ? p.mu : p.r);

  const double dt = p.T / static_cast<double>(p.steps);
  const double drift = (drift_rate - 0.5 * p.sigma * p.sigma) * dt;
  const double vol = p.sigma * std::sqrt(dt);

  const std::size_t n = static_cast<std::size_t>(p.steps) + 1;
  std::vector<std::vector<double>> sum_t(static_cast<std::size_t>(threads), std::vector<double>(n, 0.0));
  std::vector<std::vector<double>> sumsq_t(static_cast<std::size_t>(threads), std::vector<double>(n, 0.0));

  auto t0 = std::chrono::high_resolution_clock::now();

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    auto& sum = sum_t[static_cast<std::size_t>(tid)];
    auto& sumsq = sumsq_t[static_cast<std::size_t>(tid)];

    // decorrelate RNG per thread
    std::uint64_t seed = p.seed ^ (0x9E3779B97F4A7C15ULL * static_cast<std::uint64_t>(tid + 1));
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);

#pragma omp for schedule(static)
    for (std::int64_t path = 0; path < p.paths; ++path) {
      (void)path;
      double S = p.S0;
      sum[0] += S;
      sumsq[0] += S * S;
      for (int s = 1; s <= p.steps; ++s) {
        const double z = nd(rng);
        S *= std::exp(drift + vol * z);
        sum[static_cast<std::size_t>(s)] += S;
        sumsq[static_cast<std::size_t>(s)] += S * S;
      }
    }
  }

  // Reduce across threads.
  std::vector<double> sum(n, 0.0), sumsq(n, 0.0);
  for (int tid = 0; tid < threads; ++tid) {
    for (std::size_t i = 0; i < n; ++i) {
      sum[i] += sum_t[static_cast<std::size_t>(tid)][i];
      sumsq[i] += sumsq_t[static_cast<std::size_t>(tid)][i];
    }
  }

  std::vector<double> mean(n, 0.0), stdev(n, 0.0);
  const double dn = static_cast<double>(p.paths);
  for (std::size_t i = 0; i < n; ++i) {
    const double m = sum[i] / dn;
    const double ex2 = sumsq[i] / dn;
    const double var = std::max(ex2 - m * m, 0.0);
    mean[i] = m;
    stdev[i] = std::sqrt(var);
  }

  write_avg_path_csv(p.avg_path_csv, p.steps, p.T, mean, stdev);

  auto t1 = std::chrono::high_resolution_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "CPU(OpenMP) threads=" << threads << " paths=" << p.paths << " steps=" << p.steps
            << " time_ms=" << ms << "\n";
  std::cout << "Wrote avg-path CSV: " << p.avg_path_csv << "\n";

  write_bench_csv(p, have_yahoo ? &ys : nullptr, have_yahoo, ms);
  if (!p.out_csv.empty()) {
    std::cout << "Appended bench CSV: " << p.out_csv << "\n";
  }

  return 0;
}


