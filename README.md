# Parallel Scalability Analysis of Monte Carlo-Based Gold Investment Risk Assessment Using OpenMP and MPI

A reproducible high-performance computing study that accelerates Monte Carlo simulation for gold-investment risk analysis using two parallel programming models:

- **OpenMP** for shared-memory multithreading
- **MPI** for distributed-memory multiprocessing

The project calibrates a Geometric Brownian Motion (GBM) model from local historical gold-price and U.S. risk-free-rate datasets, then executes large-scale simulations to estimate terminal-price distributions and tail-risk metrics. The implementation is designed to show both the **parallelization strategy** and its **measured performance impact**.

Full report: [MonteCarlo_Parallelization_Report (PDF)](result/MonteCarlo_Parallelization_Report.pdf)

> **Focus:** numerical modelling, parallel systems, reproducible experimentation, and quantitative risk analysis.

---

## Highlights

- Evaluates **10,000**, **100,000**, **1,000,000**, **10,000,000**, and **100,000,000** independent GBM paths, each consisting of **252 GBM time steps**.
- Uses **1,000,000 paths** as the primary benchmark workload: approximately **252 million stochastic state updates** per run.
- Implements functionally equivalent **OpenMP** and **MPI** backends in modern C++17.
- Tests **1, 2, 4, 8, 12, and 16 workers** on the same single-node multicore system.
- Quantifies execution time, speedup, parallel efficiency, workload scalability, and execution-time breakdown.
- Includes execution-time breakdown analysis that separates initialization, computation, and synchronization/communication overhead.
- Computes mean, standard deviation, probability of loss, average drawdown, percentiles, **VaR (95% and 99%)**, and approximate **CVaR (95%)**.
- Uses memory-efficient **10,000-bin local histograms** rather than storing all terminal prices.
- Demonstrates that synchronization overhead dominates small workloads, while computation dominates large workloads, consistent with Amdahl's and Gustafson's scaling principles.

---

## Research Objective

Monte Carlo risk simulation is naturally parallel because individual paths are independent. However, large path counts and repeated GBM time-step updates make sequential execution expensive.

The study addresses three practical questions:

1. How a path-independent Monte Carlo simulation can be partitioned safely across threads or processes.
2. How OpenMP and MPI scale under identical worker configurations.
3. How workload size affects parallel overhead, speedup, and efficiency.
4. Whether parallel execution preserves convergence toward the analytical GBM expectation while producing risk metrics efficiently.
5. How execution time is distributed among initialization, computation, and synchronization phases.

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

| Workload | Best OpenMP | Best MPI  | Observation                        |
| -------- | ----------- | --------- | ---------------------------------- |
| 10K      | 0.045 s     | 0.110 s   | MPI overhead dominates             |
| 100K     | 0.334 s     | 0.394 s   | OpenMP maintains higher efficiency |
| 1M       | 3.180 s     | 3.297 s   | Primary benchmark                  |
| 10M      | 31.777 s    | 32.044 s  | Comparable performance             |
| 100M     | 314.620 s   | 279.020 s | MPI achieves highest speedup       |

Detailed benchmark results are available in the accompanying report.

### Interpretation

- Both implementations achieve near-linear speedup up to four workers.
- Speedup gradually saturates beyond four workers due to synchronization overhead and resource contention.
- Small workloads (10K–100K) are increasingly dominated by parallel overhead, particularly in MPI.
- As workload size increases, computation increasingly dominates execution time, allowing MPI and OpenMP to exhibit comparable scalability.
- The observed scaling trends are consistent with Amdahl's Law and Gustafson's Law.

| Representative evidence |
|---|
| ![OpenMP benchmark with 16 threads](result/result_OpenMP/1M/result_OpenMP_1M_parallel_16.jpg) |
| ![MPI timing breakdown with 16 ranks](result/result_Breakdown/10K/result_Breakdown__MPI_10K_16) |

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
│   ├── main_OpenMP.exe              # OpenMP executable
│   ├── main_MPI.cpp                 # Distributed-memory implementation
│   ├── main_MPI.exe                 # MPI executable
│   ├── breakdown_OpenMP.cpp         # OpenMP timing breakdown implementation
│   ├── breakdown_OpenMP.exe         # OpenMP executable timing breakdown
│   ├── breakdown_MPI.cpp            # MPI timing breakdown implementation
│   ├── breakdown_MPI.exe            # MPI executable timing breakdown
│   ├── GoldPrice-USD.csv            # Historical gold-price data
│   └── RiskFreeRateUSA.csv          # Historical U.S. risk-free-rate data
├── result/
│   ├── result_OpenMP/
│   │   ├── 10K/                     # OpenMP outputs: 10,000 paths
│   │   ├── 100K/                    # OpenMP outputs: 100,000 paths
│   │   ├── 1M/                      # OpenMP outputs: 1,000,000 paths
│   │   ├── 10M/                     # OpenMP outputs: 10,000,000 paths
│   │   └── 100M/                    # OpenMP outputs: 100,000,000 paths
│   ├── result_MPI/
│   │   ├── 10K/                     # MPI outputs: 10,000 paths
│   │   ├── 100K/                    # MPI outputs: 100,000 paths
│   │   ├── 1M/                      # MPI outputs: 1,000,000 paths
│   │   ├── 10M/                     # MPI outputs: 10,000,000 paths
│   │   └── 100M/                    # MPI outputs: 100,000,000 paths
│   ├── result_Breakdown/
│   │   ├── 10K/                     # Output breakdown: 10,000 paths
│   │   ├── 1M/                      # Output breakdown: 1,000,000 paths
│   │   └── 100M/                    # Output breakdown: 100,000,000 paths
│   └── MonteCarlo_Parallelization_Report.pdf
├── README.md
└── build_requirements.txt
```

---

## Requirements

This is a **native C++ project**. See [`build_requirements.txt`](build_requirements.txt) for the system-level environment manifest.

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

> **Note:** The repository includes precompiled Windows executables (`montecarlo_openmp.exe` and `montecarlo_mpi.exe`) for convenience. Linux and macOS users should rebuild the executables using the build commands below.

---

## Build (Optional)

Precompiled Windows executables are already included in the `src/` directory. Rebuild only if you modify the source code.

### OpenMP

```bash
g++ -O3 -std=c++17 -fopenmp src/main_OpenMP.cpp -o src/montecarlo_openmp
```

### MPI

```bash
mpicxx -O3 -std=c++17 src/main_MPI.cpp -o src/montecarlo_mpi
```

For Windows command prompts, append `.exe` to output names where appropriate.

---

## Run

### OpenMP

Set the desired number of threads, then execute the model.

**Bash / Linux / macOS**

```bash
export OMP_NUM_THREADS=8
./src/montecarlo_openmp 1000000 src/GoldPrice-USD.csv src/RiskFreeRateUSA.csv
```

**PowerShell**

```powershell
$env:OMP_NUM_THREADS = 8
.\src\montecarlo_openmp.exe 1000000 src\GoldPrice-USD.csv src\RiskFreeRateUSA.csv
```

### MPI

```bash
mpirun -np 8 ./src/montecarlo_mpi 1000000 src/GoldPrice-USD.csv src/RiskFreeRateUSA.csv
```

Some Open MPI installations require:

```bash
mpirun --oversubscribe -np 8 ./src/montecarlo_mpi 1000000 src/GoldPrice-USD.csv src/RiskFreeRateUSA.csv
```

On Windows environments using Microsoft MPI, the equivalent launcher is commonly:

```powershell
mpiexec -n 8 .\src\montecarlo_mpi.exe 1000000 src\GoldPrice-USD.csv src\RiskFreeRateUSA.csv
```

### Command-line arguments

```text
<executable> [simulation_count] [gold_price_csv] [risk_free_rate_csv]
```

Example:

```bash
./src/montecarlo_openmp 500000 src/GoldPrice-USD.csv src/RiskFreeRateUSA.csv
```

---

## Reproducibility Notes

- The MPI implementation derives deterministic, rank-specific random seeds from a fixed base seed, allowing repeatable MPI benchmark behavior for a fixed environment.
- The OpenMP implementation creates independent thread-level random streams using time and thread identifiers. Its numerical outputs therefore vary slightly from run to run, as expected for Monte Carlo simulation.
- The included benchmark screenshots document a particular machine and toolchain. Before using the numbers in a report or publication, record the CPU model, core count, memory, OS, compiler version, compiler flags, MPI distribution, and number of repeated runs.
- The 10,000-bin histogram trades a small quantile approximation error for bounded memory use. This is deliberate: storing every terminal price would scale memory linearly with the number of paths.
- Additional breakdown experiments instrument the execution into initialization, computation, and synchronization phases to quantify parallel overhead under representative workloads.

---

## Limitations and Future Work

This project is intentionally focused on CPU-based parallel Monte Carlo acceleration. High-value extensions include:

- evaluate hybrid MPI + OpenMP on multi-socket systems;
- extend to multi-node cluster environments;
- compare CPU implementations with OpenCL, CUDA, or HIP GPU acceleration;
- investigate adaptive scheduling and load balancing for heterogeneous workloads;
- replace histogram-based quantiles with exact distributed selection algorithms.

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
