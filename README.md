# Parallel Acceleration of Monte Carlo–Based Gold Investment Risk Analysis

A reproducible high-performance computing study that accelerates Monte Carlo simulation for gold-investment risk analysis using two parallel programming models:

- **OpenMP** for shared-memory multithreading
- **MPI** for distributed-memory multiprocessing

The project calibrates a Geometric Brownian Motion (GBM) model from local historical gold-price and U.S. risk-free-rate datasets, then executes large-scale simulations to estimate terminal-price distributions and tail-risk metrics. The implementation is designed to show both the **parallelization strategy** and its **measured performance impact**.

Full report: [MonteCarlo_Parallelization_Report (PDF)](result/MonteCarlo_Parallelization_Report.pdf)

> **Focus:** numerical modelling, parallel systems, reproducible experimentation, and quantitative risk analysis.

---

## Highlights

- Evaluates **100,000**, **1,000,000**, and **10,000,000** independent GBM paths, with **252 trading-day steps per path**.
- Uses **1,000,000 paths** as the primary benchmark workload: approximately **252 million stochastic state updates** per run.
- Implements functionally equivalent **OpenMP** and **MPI** backends in modern C++17.
- Tests **1, 2, 4, 8, 12, and 16 workers** on the same single-node multicore system.
- Quantifies execution time, speedup, parallel efficiency, and workload-size scalability.
- Computes mean, standard deviation, probability of loss, average drawdown, percentiles, **VaR (95% and 99%)**, and approximate **CVaR (95%)**.
- Uses memory-efficient **10,000-bin local histograms** rather than storing all terminal prices.
- Shows that MPI overhead is more visible for the 100,000-path workload but is amortized as the simulation workload grows.

---

## Research Objective

Monte Carlo risk simulation is naturally parallel because individual paths are independent. However, large path counts and repeated GBM time-step updates make sequential execution expensive.

The study addresses three practical questions:

1. How a path-independent Monte Carlo simulation can be partitioned safely across threads or processes.
2. How OpenMP and MPI scale under identical worker configurations.
3. How workload size affects parallel overhead, speedup, and efficiency.
4. Whether parallel execution preserves convergence toward the analytical GBM expectation while producing risk metrics efficiently.

---

## Model and Risk Metrics

### Geometric Brownian Motion

The simulated gold price follows:

```math
S_{t+\Delta t} = S_t \exp\left[\left(r - \frac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t}Z\right]
```

where:

- `S₀` is the latest historical gold price,
- `r` is the latest annual risk-free rate,
- `σ` is annualized historical volatility from daily log returns,
- `Δt = 1 / 252`, and
- `Z ~ N(0, 1)`.

For one-year validation, the simulation mean is compared with the GBM analytical expectation:

```math
\mathbb{E}[S_T] = S_0e^{rT}
```

### Reported outputs

| Category | Metrics |
|---|---|
| Distribution | mean, standard deviation, minimum, maximum |
| Downside exposure | loss probability, average drawdown |
| Percentiles | P1, P5, P10, P25, P50, P75, P90, P95 |
| Tail risk | VaR 95%, VaR 99%, approximate CVaR 95% |

---

## Parallelization Design

### OpenMP: shared-memory parallelism

The OpenMP implementation parallelizes the outer Monte Carlo path loop. Each thread owns its random-number generator and local histogram, while scalar statistics are combined with OpenMP reductions. Per-thread histograms avoid race conditions during bin updates and are merged after the parallel region.

```text
Historical CSV data
        │
        ▼
GBM calibration (S₀, r, σ)
        │
        ▼
OpenMP parallel-for over Monte Carlo paths
        │
        ├── Thread-local RNG
        ├── Thread-local histogram
        └── Reduction of scalar statistics
        │
        ▼
Merged distribution and risk metrics
```

### MPI: distributed-memory parallelism

MPI rank 0 reads and calibrates the input data, broadcasts the model parameters, and assigns a balanced contiguous block of simulation paths to each rank. Each rank builds local statistics and a local histogram. Global statistics are formed with `MPI_Reduce`; the reported execution time is the maximum elapsed time across ranks.

```text
Rank 0: read CSV + calibrate GBM
        │
        ▼  MPI_Bcast(S₀, r, σ, N)
All ranks simulate disjoint path blocks
        │
        ├── Local RNG and path simulation
        ├── Local histogram
        └── Local statistics
        │
        ▼  MPI_Reduce
Rank 0: global percentiles, VaR, CVaR, and timing
```

---

## Benchmark Results

All experiments use the same 252-step GBM model and worker configurations of 1, 2, 4, 8, 12, and 16. Results are specific to the recorded machine and toolchain and should not be treated as universal performance claims.

### Workload-Size Scalability Summary

| Workload | Best OpenMP result | Best MPI result | Main observation |
|---:|---|---|---|
| 100,000 paths | 0.334 s, 8.41× speedup, 16 threads | 0.394 s, 7.19× speedup, 16 ranks | MPI overhead is more visible at high worker counts |
| 1,000,000 paths | 3.180 s, 8.88× speedup, 16 threads | 3.297 s, 8.60× speedup, 16 ranks | Primary benchmark; both implementations scale strongly |
| 10,000,000 paths | 31.777 s, 8.92× speedup, 16 threads | 32.044 s, 8.85× speedup, 16 ranks | OpenMP and MPI become nearly identical as overhead is amortized |

### 100,000-Path Workload

| Workers | OpenMP time (s) | MPI time (s) | OpenMP speedup | MPI speedup | OpenMP efficiency | MPI efficiency |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.809 | 2.834 | 1.00× | 1.00× | 100.00% | 100.00% |
| 2 | 1.374 | 1.357 | 2.04× | 2.09× | 102.22% | 104.42% |
| 4 | 0.679 | 0.720 | 4.14× | 3.94× | 103.42% | 98.40% |
| 8 | 0.515 | 0.565 | 5.45× | 5.02× | 68.18% | 62.70% |
| 12 | 0.404 | 0.425 | 6.95× | 6.67× | 57.94% | 55.57% |
| 16 | 0.334 | 0.394 | **8.41×** | **7.19×** | 52.56% | 44.96% |

### 1,000,000-Path Primary Benchmark

| Workers | OpenMP time (s) | MPI time (s) | OpenMP speedup | MPI speedup | OpenMP efficiency | MPI efficiency |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 28.220 | 28.360 | 1.00× | 1.00× | 100.00% | 100.00% |
| 2 | 13.450 | 13.430 | 2.10× | 2.11× | 104.87% | 105.60% |
| 4 | 6.780 | 7.400 | 4.16× | 3.84× | 104.01% | 95.88% |
| 8 | 4.880 | 5.440 | 5.78× | 5.21× | 72.30% | 65.14% |
| 12 | 3.680 | 3.520 | 7.66× | 8.05× | 63.86% | 67.10% |
| 16 | 3.180 | 3.300 | **8.88×** | **8.60×** | 55.47% | 53.76% |

### 10,000,000-Path Workload

| Workers | OpenMP time (s) | MPI time (s) | OpenMP speedup | MPI speedup | OpenMP efficiency | MPI efficiency |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 283.516 | 283.519 | 1.00× | 1.00× | 100.00% | 100.00% |
| 2 | 134.355 | 134.485 | 2.11× | 2.11× | 105.51% | 105.41% |
| 4 | 67.423 | 67.393 | 4.21× | 4.21× | 105.13% | 105.17% |
| 8 | 47.676 | 48.257 | 5.95× | 5.88× | 74.33% | 73.44% |
| 12 | 35.419 | 36.038 | 8.00× | 7.87× | 66.71% | 65.56% |
| 16 | 31.777 | 32.044 | **8.92×** | **8.85×** | 55.76% | 55.30% |

### Interpretation

- Both implementations show near-linear speedup through four workers.
- Efficiency decreases beyond four workers because of synchronization, reduction, memory contention, operating-system scheduling, and use of logical processors beyond the 12 physical cores.
- At 100,000 paths, MPI efficiency falls to **44.96%** at 16 ranks, compared with **52.56%** for OpenMP, because fixed process-management and reduction overhead represent a larger fraction of runtime.
- At 10,000,000 paths, the larger computation per worker amortizes MPI overhead. OpenMP and MPI reach nearly identical 16-worker speedups of **8.92×** and **8.85×**, respectively.
- Small efficiency values above 100% at two or four workers are attributed to cache effects, CPU turbo behavior, and measurement variation rather than true superlinear scaling.

| Representative evidence |
|---|
| ![OpenMP benchmark with 16 threads](result/result_OpenMP/1M/result_OpenMP_1M_parallel_16.jpg) |
| ![MPI benchmark with 16 ranks](result/result_MPI/1M/result_MPI_1M_parallel_16.jpg) |

---

## Numerical Verification and Risk Metrics

For the 1,000,000-path primary benchmark, both implementations produced mean terminal prices close to the analytical GBM expectation of **$4,687.23**.

| Metric | OpenMP (16 threads) | MPI (16 ranks) |
|---|---:|---:|
| Execution time (s) | 3.180 | 3.297 |
| Simulated mean price | $4,687.61 | $4,685.39 |
| Analytical mean price | $4,687.23 | $4,687.23 |
| Relative error | 0.01% | 0.04% |
| Final price standard deviation | $832.44 | $832.68 |
| Probability of loss | 43.24% | 43.52% |
| Average drawdown | 10.91% | 10.93% |
| VaR 95% | $1,029.08 | $1,030.57 |
| VaR 99% | $1,419.46 | $1,422.44 |
| Approximate CVaR 95% | $1,267.71 | $1,269.74 |

The results preserve the expected statistical behavior of the GBM model while providing efficient approximations of terminal-price percentiles and tail-risk metrics.

---

## Repository Structure

```text
MonteCarlo_HPC_Parallelization/
├── src/
│   ├── main_OpenMP.cpp              # Shared-memory implementation
│   ├── main_MPI.cpp                 # Distributed-memory implementation
│   ├── GoldPrice-USD.csv            # Historical gold-price data
│   └── RiskFreeRateUSA.csv          # Historical U.S. risk-free-rate data
├── result/
│   ├── result_OpenMP/
│   │   ├── 100K/                    # OpenMP outputs: 100,000 paths
│   │   ├── 1M/                      # OpenMP outputs: 1,000,000 paths
│   │   └── 10M/                     # OpenMP outputs: 10,000,000 paths
│   ├── result_MPI/
│   │   ├── 100K/                    # MPI outputs: 100,000 paths
│   │   ├── 1M/                      # MPI outputs: 1,000,000 paths
│   │   └── 10M/                     # MPI outputs: 10,000,000 paths
│   └── MonteCarlo_Parallelization_Report.pdf
├── README.md
└── requirements.txt
```

---

## Requirements

This is a **native C++ project**. See [`requirements.txt`](requirements.txt) for the system-level environment manifest.

Minimum environment:

- C++17-compatible compiler: GCC, Clang, or MSVC
- OpenMP runtime and compiler support
- MPI implementation for the MPI executable: Open MPI or MPICH
- A terminal environment with `g++` and `mpicxx`/`mpic++` available on `PATH`

### Linux (Ubuntu/Debian example)

```bash
sudo apt update
sudo apt install build-essential openmpi-bin libopenmpi-dev
```

### macOS (Homebrew example)

```bash
brew install gcc open-mpi
```

### Windows

Use one consistent native toolchain environment, such as **MSYS2 UCRT64 + Open MPI**, rather than Git Bash alone. Ensure that `g++`, `mpicxx`, and `mpirun` or `mpiexec` are visible from the same shell before compiling.

---

## Build

Run the following from the repository root.

### OpenMP

```bash
mkdir -p bin
g++ -O3 -std=c++17 -fopenmp src/main_OpenMP.cpp -o bin/montecarlo_openmp
```

### MPI

```bash
mkdir -p bin
mpicxx -O3 -std=c++17 src/main_MPI.cpp -o bin/montecarlo_mpi
```

For Windows command prompts, append `.exe` to output names where appropriate.

---

## Run

### OpenMP

Set the desired number of threads, then execute the model.

**Bash / Linux / macOS**

```bash
export OMP_NUM_THREADS=8
./bin/montecarlo_openmp 1000000 src/GoldPrice-USD.csv src/RiskFreeRateUSA.csv
```

**PowerShell**

```powershell
$env:OMP_NUM_THREADS = 8
.\bin\montecarlo_openmp.exe 1000000 src\GoldPrice-USD.csv src\RiskFreeRateUSA.csv
```

### MPI

```bash
mpirun -np 8 ./bin/montecarlo_mpi 1000000 src/GoldPrice-USD.csv src/RiskFreeRateUSA.csv
```

Some Open MPI installations require:

```bash
mpirun --oversubscribe -np 8 ./bin/montecarlo_mpi 1000000 src/GoldPrice-USD.csv src/RiskFreeRateUSA.csv
```

On Windows environments using Microsoft MPI, the equivalent launcher is commonly:

```powershell
mpiexec -n 8 .\bin\montecarlo_mpi.exe 1000000 src\GoldPrice-USD.csv src\RiskFreeRateUSA.csv
```

### Command-line arguments

```text
<executable> [simulation_count] [gold_price_csv] [risk_free_rate_csv]
```

Example:

```bash
./bin/montecarlo_openmp 500000 src/GoldPrice-USD.csv src/RiskFreeRateUSA.csv
```

---

## Reproducibility Notes

- The MPI implementation derives deterministic, rank-specific random seeds from a fixed base seed, allowing repeatable MPI benchmark behavior for a fixed environment.
- The OpenMP implementation creates independent thread-level random streams using time and thread identifiers. Its numerical outputs therefore vary slightly from run to run, as expected for Monte Carlo simulation.
- The included benchmark screenshots document a particular machine and toolchain. Before using the numbers in a report or publication, record the CPU model, core count, memory, OS, compiler version, compiler flags, MPI distribution, and number of repeated runs.
- The 10,000-bin histogram trades a small quantile approximation error for bounded memory use. This is deliberate: storing every terminal price would scale memory linearly with the number of paths.

---

## Limitations and Future Work

This project is intentionally focused on CPU-based parallel Monte Carlo acceleration. High-value extensions include:

- add a `CMakeLists.txt` and CI workflow for one-command cross-platform builds;
- publish data provenance, retrieval dates, and dataset licensing metadata;
- run repeated trials with median runtime, standard deviation, efficiency, and confidence intervals;
- compare static, dynamic, and guided scheduling in OpenMP;
- add hybrid MPI + OpenMP and GPU/CUDA variants;
- replace histogram quantiles with distributed selection or streaming quantile algorithms for tighter tail-risk accuracy;
- add unit tests for CSV parsing, GBM calibration, reductions, and risk-metric calculations.

---

## Authors

- [Andhika Narawangsa Susilo](https://github.com/andhikanarawangsa)
- [Jovan Hosea H. Napitupulu](https://github.com/JovanHosea/)
- Kayla Pramudio Bagas Aryasatya

---

## Citation

If this repository contributes to academic work, cite it as:

```bibtex
@software{montecarlo_hpc_parallelization,
  author  = {Susilo, Andhika Narawangsa and Napitupulu, Jovan Hosea H. and Aryasatya, Kayla Pramudio Bagas},
  title   = {Parallel Acceleration of Monte Carlo-Based Gold Investment Risk Analysis Using OpenMP and MPI},
  year    = {2026},
  url     = {https://github.com/andhikanarawangsa/MonteCarlo_HPC_Parallelization}
}
```

---

## Disclaimer

This repository is an academic and technical demonstration of high-performance Monte Carlo methods. It is **not financial advice** and should not be used as the sole basis for investment decisions.
