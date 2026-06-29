// Monte Carlo Simulation Parallelization Using MPI
// 1,000,000 paths, 252 GBM steps per path, histogram-based risk metrics.
// 13222036 - Andhika Narawangsa Susilo
// 13622005 - Jovan Hosea H Napitupulu
// 13222117 - Kayla Pramudio Bagas Aryasatya

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// ============================================================
// CSV reader
// ============================================================
std::vector<double> readCSVColumn(const std::string& filename, int colIndex) {
    std::vector<double> data;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot Open File: " << filename << '\n';
        return data;
    }

    std::getline(file, line); // skip header
    const char delim = ';';

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        int currentCol = 0;

        while (std::getline(ss, token, delim)) {
            if (currentCol == colIndex) {
                try {
                    // erase sign
                    token.erase(std::remove(token.begin(), token.end(), ','), token.end());
                    data.push_back(std::stod(token));
                } catch (...) {
                    // ignore invalid data line
                }
                break;
            }
            ++currentCol;
        }
    }

    return data;
}

// ============================================================
// Historical volatility based on daily log returns
// ============================================================
double calculateHistoricalVolatility(const std::vector<double>& prices) {
    if (prices.size() < 2) return 0.0;

    std::vector<double> logReturns;
    logReturns.reserve(prices.size() - 1);

    double sumReturns = 0.0;
    for (std::size_t i = 1; i < prices.size(); ++i) {
        if (prices[i - 1] <= 0.0 || prices[i] <= 0.0) continue;

        const double ret = std::log(prices[i] / prices[i - 1]);
        logReturns.push_back(ret);
        sumReturns += ret;
    }

    if (logReturns.size() < 2) return 0.0;

    const double mean = sumReturns / static_cast<double>(logReturns.size());
    double variance = 0.0;

    for (double r : logReturns) {
        variance += (r - mean) * (r - mean);
    }

    variance /= static_cast<double>(logReturns.size() - 1);
    return std::sqrt(variance) * std::sqrt(252.0);
}

// ============================================================
// Histogram helpers
// ============================================================
static const int N_BINS = 10000;

int priceToBin(double price, double pMin, double pMax) {
    if (price <= pMin) return 0;
    if (price >= pMax) return N_BINS - 1;

    const int bin = static_cast<int>((price - pMin) / (pMax - pMin) * N_BINS);
    return std::max(0, std::min(N_BINS - 1, bin));
}

double quantileFromHistogram(const std::vector<long long>& hist,
                             double pMin,
                             double pMax,
                             double quantile,
                             long long total) {
    if (total <= 0) return 0.0;

    long long target = static_cast<long long>(std::ceil(quantile * total));
    target = std::max(1LL, target);

    long long cumulative = 0;
    const double binWidth = (pMax - pMin) / static_cast<double>(N_BINS);

    for (int b = 0; b < N_BINS; ++b) {
        cumulative += hist[b];
        if (cumulative >= target) {
            return pMin + (b + 0.5) * binWidth;
        }
    }

    return pMax;
}

double expectedShortfallFromHistogram(const std::vector<long long>& hist,
                                      double pMin,
                                      double pMax,
                                      double quantile,
                                      long long total) {
    if (total <= 0) return 0.0;

    long long target = static_cast<long long>(std::ceil(quantile * total));
    target = std::max(1LL, target);

    const double binWidth = (pMax - pMin) / static_cast<double>(N_BINS);
    long long taken = 0;
    double sum = 0.0;

    for (int b = 0; b < N_BINS && taken < target; ++b) {
        const long long canTake = std::min(hist[b], target - taken);
        const double binCenter = pMin + (b + 0.5) * binWidth;

        sum += static_cast<double>(canTake) * binCenter;
        taken += canTake;
    }

    return (taken > 0) ? sum / static_cast<double>(taken) : 0.0;
}

// ============================================================
// Main: MPI-only Monte Carlo GBM
// ============================================================
int main(int argc, char** argv) {
    // Start before MPI initialization so the reported runtime includes MPI runtime
    // initialization, workload setup, RNG setup, the initial synchronization, and
    // the subsequent simulation/reduction phase.
    const auto programStartTime = std::chrono::steady_clock::now();

    MPI_Init(&argc, &argv);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    long long totalSimulations = 1000000LL;
    std::string pathEmas = "GoldPrice-USD.csv";
    std::string pathRate = "RiskFreeRateUSA.csv";

    int inputOk = 1;
    double S0 = 0.0;
    double r = 0.0;
    double sigma = 0.0;

    const double T = 1.0;
    const int N_STEPS = 252;
    const double HIST_MIN = 100.0;
    const double HIST_MAX = 15000.0;

    // Rank 0 alone reads the CSV files and calibrates GBM parameters.
    if (rank == 0) {
        if (argc >= 2) {
            try {
                totalSimulations = std::stoll(argv[1]);
            } catch (...) {
                std::cerr << "ERROR: Simulation Argument Invalid.\n";
                inputOk = 0;
            }
        }

        if (argc >= 3) pathEmas = argv[2];
        if (argc >= 4) pathRate = argv[3];

        if (totalSimulations <= 0) {
            std::cerr << "ERROR: Simulation Count Must be > 0.\n";
            inputOk = 0;
        }

        if (inputOk) {
            std::vector<double> goldPrices = readCSVColumn(pathEmas, 1);
            std::vector<double> riskFreeRates = readCSVColumn(pathRate, 1);

            if (goldPrices.empty() || riskFreeRates.empty()) {
                std::cerr << "ERROR: Gold Price or Risk Free Rate File is Empty.\n";
                inputOk = 0;
            } else {
                std::reverse(goldPrices.begin(), goldPrices.end());
                S0 = goldPrices.back();
                r = riskFreeRates.back() / 100.0;
                sigma = calculateHistoricalVolatility(goldPrices);

                std::cout << std::fixed << std::setprecision(6);
                std::cout << "=== REAL WORLD HISTORY DATA CALIBRATION===\n";
                std::cout << "Gold Price Days Count: " << goldPrices.size() << " days\n";
                std::cout << "Start Price (S0)        : $" << S0 << "\n";
                std::cout << "Risk-Free Rate (r)     : " << (r * 100.0) << "%\n";
                std::cout << "History Volatility   : " << (sigma * 100.0) << "%\n\n";
                std::cout << "Start Execution " << totalSimulations
                          << " Monte Carlo MPI (252-step GBM)...\n";
                std::cout << "MPI ranks count       : " << worldSize << "\n\n";
            }
        }
    }

    // All ranks must receive a consistent success/failure status before continuing.
    MPI_Bcast(&inputOk, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!inputOk) {
        MPI_Finalize();
        return 1;
    }

    // Broadcast calibration parameters to all MPI ranks.
    MPI_Bcast(&totalSimulations, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&S0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&r, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sigma, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Static and balanced path distribution: every simulation is assigned exactly once.
    const long long baseWork = totalSimulations / worldSize;
    const long long remainder = totalSimulations % worldSize;
    const long long localSimulations = baseWork + (rank < remainder ? 1 : 0);
    const long long localStartIndex = rank * baseWork + std::min<long long>(rank, remainder);
    (void)localStartIndex; // Kept for traceability and possible future path-index RNG.

    const double dt = T / static_cast<double>(N_STEPS);
    const double drift = (r - 0.5 * sigma * sigma) * dt;
    const double vol = sigma * std::sqrt(dt);

    // Each rank owns independent local statistics and a local histogram.
    double localSumPrice = 0.0;
    double localSumSqPrice = 0.0;
    long long localLossCount = 0;
    double localMinPrice = std::numeric_limits<double>::max();
    double localMaxPrice = std::numeric_limits<double>::lowest();
    double localSumDrawdown = 0.0;
    std::vector<long long> localHist(N_BINS, 0);

    // Deterministic but distinct seed per rank: reproducible across benchmark runs.
    const unsigned long long baseSeed = 20260620ULL;
    std::seed_seq seedSeq{
        static_cast<unsigned int>(baseSeed),
        static_cast<unsigned int>(baseSeed >> 32),
        static_cast<unsigned int>(rank),
        static_cast<unsigned int>(worldSize),
        static_cast<unsigned int>(totalSimulations & 0xffffffffULL)
    };
    std::mt19937_64 generator(seedSeq);
    std::normal_distribution<double> normalDist(0.0, 1.0);

    // Keep the initial synchronization inside the measured interval.
    MPI_Barrier(MPI_COMM_WORLD);

    for (long long i = 0; i < localSimulations; ++i) {
        double path = S0;
        double pathMax = S0;

        for (int step = 0; step < N_STEPS; ++step) {
            const double Z = normalDist(generator);
            path *= std::exp(drift + vol * Z);
            pathMax = std::max(pathMax, path);
        }

        const double ST = path;
        localSumPrice += ST;
        localSumSqPrice += ST * ST;
        if (ST < S0) ++localLossCount;

        localMinPrice = std::min(localMinPrice, ST);
        localMaxPrice = std::max(localMaxPrice, ST);
        localSumDrawdown += (pathMax - ST) / pathMax;

        const int bin = priceToBin(ST, HIST_MIN, HIST_MAX);
        ++localHist[bin];
    }

    // Global reductions: rank 0 obtains the complete result.
    double globalSumPrice = 0.0;
    double globalSumSqPrice = 0.0;
    long long globalLossCount = 0;
    double globalMinPrice = 0.0;
    double globalMaxPrice = 0.0;
    double globalSumDrawdown = 0.0;
    std::vector<long long> globalHist(N_BINS, 0);

    MPI_Reduce(&localSumPrice, &globalSumPrice, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localSumSqPrice, &globalSumSqPrice, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localLossCount, &globalLossCount, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localMinPrice, &globalMinPrice, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localMaxPrice, &globalMaxPrice, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localSumDrawdown, &globalSumDrawdown, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(localHist.data(), globalHist.data(), N_BINS,
               MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Stop after all simulation-result reductions. MPI_Reduce below selects the
    // longest elapsed time among ranks, which represents the total parallel runtime.
    const auto programEndTime = std::chrono::steady_clock::now();
    const double localElapsed = std::chrono::duration<double>(
        programEndTime - programStartTime
    ).count();

    double executionTime = 0.0;
    MPI_Reduce(&localElapsed, &executionTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double meanPrice = globalSumPrice / static_cast<double>(totalSimulations);
        double variance = (globalSumSqPrice / static_cast<double>(totalSimulations)) -
                          (meanPrice * meanPrice);
        variance = std::max(0.0, variance);

        const double stdPrice = std::sqrt(variance);
        const double lossProb = 100.0 * static_cast<double>(globalLossCount) /
                                static_cast<double>(totalSimulations);
        const double avgDrawdown = 100.0 * globalSumDrawdown /
                                   static_cast<double>(totalSimulations);
        const double analyticalMean = S0 * std::exp(r * T);

        const double p1 = quantileFromHistogram(globalHist, HIST_MIN, HIST_MAX, 0.01, totalSimulations);
        const double p5 = quantileFromHistogram(globalHist, HIST_MIN, HIST_MAX, 0.05, totalSimulations);
        const double p10 = quantileFromHistogram(globalHist, HIST_MIN, HIST_MAX, 0.10, totalSimulations);
        const double p25 = quantileFromHistogram(globalHist, HIST_MIN, HIST_MAX, 0.25, totalSimulations);
        const double p50 = quantileFromHistogram(globalHist, HIST_MIN, HIST_MAX, 0.50, totalSimulations);
        const double p75 = quantileFromHistogram(globalHist, HIST_MIN, HIST_MAX, 0.75, totalSimulations);
        const double p90 = quantileFromHistogram(globalHist, HIST_MIN, HIST_MAX, 0.90, totalSimulations);
        const double p95 = quantileFromHistogram(globalHist, HIST_MIN, HIST_MAX, 0.95, totalSimulations);

        const double VaR95 = S0 - p5;
        const double VaR99 = S0 - p1;
        const double esPrice5 = expectedShortfallFromHistogram(
            globalHist, HIST_MIN, HIST_MAX, 0.05, totalSimulations
        );
        const double CVaR95 = S0 - esPrice5;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "=== MONTE CARLO MPI RESULT ===\n";
        std::cout << "Simulation Count       : " << totalSimulations << "\n";
        std::cout << "Step per Path          : " << N_STEPS << "\n";
        std::cout << "Execution Time (inclusive): " << executionTime << " s\n\n";

        std::cout << "Simulation Price Mean  : $" << meanPrice << "\n";
        std::cout << "GBM Analythics Mean    : $" << analyticalMean << "\n";
        std::cout << "Final Price Std Dev    : $" << stdPrice << "\n";
        std::cout << "Final Price Min        : $" << globalMinPrice << "\n";
        std::cout << "Final Price Max        : $" << globalMaxPrice << "\n";
        std::cout << "Loss Pribability       : " << lossProb << "%\n";
        std::cout << "Average Drawdown       : " << avgDrawdown << "%\n\n";

        std::cout << "=== FINAL PRICE PERCENTILE ===\n";
        std::cout << "P1                    : $" << p1 << "\n";
        std::cout << "P5  (worst 5%)        : $" << p5 << "\n";
        std::cout << "P10 (worst 10%)       : $" << p10 << "\n";
        std::cout << "P25                   : $" << p25 << "\n";
        std::cout << "P50 (Median)          : $" << p50 << "\n";
        std::cout << "P75                   : $" << p75 << "\n";
        std::cout << "P90                   : $" << p90 << "\n";
        std::cout << "P95 (best 5%)         : $" << p95 << "\n\n";

        std::cout << "=== RISK METRICS ===\n";
        std::cout << "VaR 95%               : $" << VaR95 << "\n";
        std::cout << "VaR 99%               : $" << VaR99 << "\n";
        std::cout << "CVaR 95% approx.      : $" << CVaR95 << "\n";
    }

    MPI_Finalize();
    return 0;
}
