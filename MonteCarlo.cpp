// Monte Carlo Simulation Parallelization
// 13222036 - Andhika Narawangsa Susilo
// 13222117 - Kayla Pramudio Bagas Aryasatya
// 13622005 - Jovan Hosea H Napitupulu

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>
#include <chrono>
#include <iomanip>
#include <string>
#include <omp.h>

// ============================================================
//  CSV reader
// ============================================================
std::vector<double> readCSVColumn(const std::string& filename, int colIndex) {
    std::vector<double> data;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "ERROR: Tidak dapat membuka file: " << filename << std::endl;
        return data;
    }

    std::getline(file, line); // skip header
    const char delim = ';';

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        int currentCol = 0;
        double parsedValue = 0.0;
        bool found = false;

        while (std::getline(ss, token, delim)) {
            if (currentCol == colIndex) {
                try {
                    // Menghapus pemisah ribuan koma
                    token.erase(std::remove(token.begin(), token.end(), ','), token.end());
                    parsedValue = std::stod(token);
                    found = true;
                } catch (...) {
                    found = false;
                }
                break;
            }
            currentCol++;
        }

        if (found) data.push_back(parsedValue);
    }

    return data;
}

// ============================================================
//  Historical volatility berbasis log return harian
// ============================================================
double calculateHistoricalVolatility(const std::vector<double>& prices) {
    if (prices.size() < 2) return 0.0;

    std::vector<double> logReturns;
    logReturns.reserve(prices.size() - 1);

    double sumReturns = 0.0;

    for (size_t i = 1; i < prices.size(); ++i) {
        if (prices[i - 1] <= 0.0 || prices[i] <= 0.0) continue;

        double ret = std::log(prices[i] / prices[i - 1]);
        logReturns.push_back(ret);
        sumReturns += ret;
    }

    if (logReturns.size() < 2) return 0.0;

    double mean = sumReturns / static_cast<double>(logReturns.size());

    double variance = 0.0;
    for (double r : logReturns) {
        variance += (r - mean) * (r - mean);
    }

    variance /= static_cast<double>(logReturns.size() - 1);

    // Annualized volatility, asumsi 252 trading days per year
    return std::sqrt(variance) * std::sqrt(252.0);
}

// ============================================================
//  Histogram-based quantile helper
// ============================================================
static const int N_BINS = 10000;

int priceToBin(double price, double pMin, double pMax) {
    if (price <= pMin) return 0;
    if (price >= pMax) return N_BINS - 1;

    int bin = static_cast<int>((price - pMin) / (pMax - pMin) * N_BINS);

    if (bin < 0) return 0;
    if (bin >= N_BINS) return N_BINS - 1;
    return bin;
}

double quantileFromHistogram(const std::vector<long long>& hist,
                             double pMin,
                             double pMax,
                             double quantile,
                             long long total) {
    if (total <= 0) return 0.0;

    long long target = static_cast<long long>(std::ceil(quantile * total));
    if (target < 1) target = 1;

    long long cumul = 0;
    double binWidth = (pMax - pMin) / static_cast<double>(N_BINS);

    for (int b = 0; b < N_BINS; ++b) {
        cumul += hist[b];
        if (cumul >= target) {
            return pMin + (b + 0.5) * binWidth;
        }
    }

    return pMax;
}

// Expected Shortfall / CVaR aproksimasi dari histogram bawah.
double expectedShortfallFromHistogram(const std::vector<long long>& hist,
                                      double pMin,
                                      double pMax,
                                      double quantile,
                                      long long total) {
    if (total <= 0) return 0.0;

    long long target = static_cast<long long>(std::ceil(quantile * total));
    if (target < 1) target = 1;

    double binWidth = (pMax - pMin) / static_cast<double>(N_BINS);
    long long taken = 0;
    double sum = 0.0;

    for (int b = 0; b < N_BINS && taken < target; ++b) {
        long long canTake = std::min(hist[b], target - taken);
        double binCenter = pMin + (b + 0.5) * binWidth;

        sum += static_cast<double>(canTake) * binCenter;
        taken += canTake;
    }

    if (taken == 0) return 0.0;
    return sum / static_cast<double>(taken);
}

// ============================================================
//  Main: OpenMP-only Monte Carlo GBM
// ============================================================
int main(int argc, char** argv) {
    // ./monte_carlo_openmp 1000000 HargaEmasUSD.csv RiskFreeRateUSA.csv
    long long totalSimulations = 1000000000LL;
    std::string pathEmas = "HargaEmasUSD.csv";
    std::string pathRate = "RiskFreeRateUSA.csv";

    if (argc >= 2) {
        try {
            totalSimulations = std::stoll(argv[1]);
        } catch (...) {
            std::cerr << "ERROR: Argumen jumlah simulasi tidak valid.\n";
            return 1;
        }
    }

    if (argc >= 3) pathEmas = argv[2];
    if (argc >= 4) pathRate = argv[3];

    if (totalSimulations <= 0) {
        std::cerr << "ERROR: Jumlah simulasi harus > 0.\n";
        return 1;
    }

    const double T = 1.0;
    const int N_STEPS = 252;
    const double HIST_MIN = 100.0;
    const double HIST_MAX = 15000.0;

    std::vector<double> goldPrices = readCSVColumn(pathEmas, 1);
    std::vector<double> riskFreeRates = readCSVColumn(pathRate, 1);

    if (goldPrices.empty() || riskFreeRates.empty()) {
        std::cerr << "ERROR: Data harga emas atau risk-free rate kosong.\n";
        return 1;
    }

    std::reverse(goldPrices.begin(), goldPrices.end());

    double S0 = goldPrices.back();
    double r = riskFreeRates.back() / 100.0;
    double sigma = calculateHistoricalVolatility(goldPrices);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== KALIBRASI DATA HISTORIS DUNIA NYATA ===\n";
    std::cout << "Jumlah Hari Harga Emas : " << goldPrices.size() << " hari\n";
    std::cout << "Harga Awal (S0)        : $" << S0 << "\n";
    std::cout << "Risk-Free Rate (r)     : " << (r * 100.0) << "%\n";
    std::cout << "Volatilitas Historis   : " << (sigma * 100.0) << "%\n\n";

    int maxThreads = omp_get_max_threads();
    std::cout << "Memulai eksekusi " << totalSimulations
              << " Monte Carlo paralel OpenMP (252-step GBM)...\n";
    std::cout << "Jumlah thread OpenMP   : " << maxThreads << "\n\n";

    double sumPrice = 0.0;
    double sumSqPrice = 0.0;
    long long lossCount = 0;
    double minPrice = std::numeric_limits<double>::max();
    double maxPrice = std::numeric_limits<double>::lowest();
    double sumDrawdown = 0.0;

    // Histogram per-thread agar tidak terjadi race condition.
    std::vector<std::vector<long long>> threadHists(
        maxThreads,
        std::vector<long long>(N_BINS, 0)
    );

    const double dt = T / static_cast<double>(N_STEPS);
    const double drift = (r - 0.5 * sigma * sigma) * dt;
    const double vol = sigma * std::sqrt(dt);

    double startTime = omp_get_wtime();

    // Paralel OpenMP
    #pragma omp parallel reduction(+:sumPrice, sumSqPrice, lossCount, sumDrawdown) \
                         reduction(min:minPrice) reduction(max:maxPrice)
    {
        int threadId = omp_get_thread_num();

        auto now = static_cast<unsigned long long>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count()
        );

        std::seed_seq seedSeq{
            static_cast<unsigned int>(now),
            static_cast<unsigned int>(now >> 32),
            static_cast<unsigned int>(threadId * 6271 + 17),
            static_cast<unsigned int>(totalSimulations & 0xffffffffULL)
        };

        std::mt19937_64 generator(seedSeq);
        std::normal_distribution<double> normalDist(0.0, 1.0);

        std::vector<long long>& hist = threadHists[threadId];

        #pragma omp for schedule(static)
        for (long long i = 0; i < totalSimulations; ++i) {
            double path = S0;
            double pathMax = S0;

            for (int step = 0; step < N_STEPS; ++step) {
                double Z = normalDist(generator);
                path *= std::exp(drift + vol * Z);
                pathMax = std::max(pathMax, path);
            }

            double ST = path;

            sumPrice += ST;
            sumSqPrice += ST * ST;

            if (ST < S0) lossCount++;

            minPrice = std::min(minPrice, ST);
            maxPrice = std::max(maxPrice, ST);

            double drawdown = (pathMax - ST) / pathMax;
            sumDrawdown += drawdown;

            int bin = priceToBin(ST, HIST_MIN, HIST_MAX);
            hist[bin]++;
        }
    }

    double endTime = omp_get_wtime();

    // Gabungkan histogram semua thread.
    std::vector<long long> globalHist(N_BINS, 0);
    for (int t = 0; t < maxThreads; ++t) {
        for (int b = 0; b < N_BINS; ++b) {
            globalHist[b] += threadHists[t][b];
        }
    }

    double meanPrice = sumPrice / static_cast<double>(totalSimulations);
    double variance = (sumSqPrice / static_cast<double>(totalSimulations)) - (meanPrice * meanPrice);
    if (variance < 0.0) variance = 0.0;

    double stdPrice = std::sqrt(variance);
    double lossProb = 100.0 * static_cast<double>(lossCount) / static_cast<double>(totalSimulations);
    double avgDrawdown = 100.0 * sumDrawdown / static_cast<double>(totalSimulations);
    double analyticalMean = S0 * std::exp(r * T);

    double p1  = quantileFromHistogram(globalHist, HIST_MIN, HIST_MAX, 0.01, totalSimulations);
    double p5  = quantileFromHistogram(globalHist, HIST_MIN, HIST_MAX, 0.05, totalSimulations);
    double p10 = quantileFromHistogram(globalHist, HIST_MIN, HIST_MAX, 0.10, totalSimulations);
    double p25 = quantileFromHistogram(globalHist, HIST_MIN, HIST_MAX, 0.25, totalSimulations);
    double p50 = quantileFromHistogram(globalHist, HIST_MIN, HIST_MAX, 0.50, totalSimulations);
    double p75 = quantileFromHistogram(globalHist, HIST_MIN, HIST_MAX, 0.75, totalSimulations);
    double p90 = quantileFromHistogram(globalHist, HIST_MIN, HIST_MAX, 0.90, totalSimulations);
    double p95 = quantileFromHistogram(globalHist, HIST_MIN, HIST_MAX, 0.95, totalSimulations);

    double VaR95 = S0 - p5;
    double VaR99 = S0 - p1;

    double esPrice5 = expectedShortfallFromHistogram(globalHist, HIST_MIN, HIST_MAX, 0.05, totalSimulations);
    double CVaR95 = S0 - esPrice5;

    std::cout << "=== HASIL MONTE CARLO OPENMP ===\n";
    std::cout << "Jumlah Simulasi        : " << totalSimulations << "\n";
    std::cout << "Jumlah Step per Path   : " << N_STEPS << "\n";
    std::cout << "Execution Time         : " << (endTime - startTime) << " detik\n\n";

    std::cout << "Mean Harga Simulasi    : $" << meanPrice << "\n";
    std::cout << "Mean Analitik GBM      : $" << analyticalMean << "\n";
    std::cout << "Std Dev Harga Akhir    : $" << stdPrice << "\n";
    std::cout << "Min Harga Akhir        : $" << minPrice << "\n";
    std::cout << "Max Harga Akhir        : $" << maxPrice << "\n";
    std::cout << "Probabilitas Rugi      : " << lossProb << "%\n";
    std::cout << "Average Drawdown       : " << avgDrawdown << "%\n\n";

    std::cout << "=== PERCENTILE HARGA AKHIR ===\n";
    std::cout << "P1                    : $" << p1  << "\n";
    std::cout << "P5 (worst 5%)         : $" << p5  << "\n";
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

    return 0;
}
