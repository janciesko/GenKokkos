#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <Kokkos_Core.hpp>

// Basic DAXPY implementation
// Computes Y = a*X + Y where a is a scalar, X and Y are vectors
void daxpy(int n, double a, const double* x, int incx, double* y, int incy) {
    for (int i = 0; i < n; i++) {
        y[i * incy] += a * x[i * incx];
    }
}

// Vectorized version using array notation (compiler may auto-vectorize)
void daxpy_vectorized(int n, double a, const double* x, double* y) {
    for (int i = 0; i < n; i++) {
        y[i] += a * x[i];
    }
}

// Template version for different data types
template<typename T>
void axpy(int n, T a, const T* x, int incx, T* y, int incy) {
    for (int i = 0; i < n; i++) {
        y[i * incy] += a * x[i * incx];
    }
}

// STL vector version
void daxpy_stl(double a, const std::vector<double>& x, std::vector<double>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector sizes must match");
    }
    
    for (size_t i = 0; i < x.size(); i++) {
        y[i] += a * x[i];
    }
}

// Kokkos version using parallel_for
void daxpy_kokkos(int n, double a, const double* x, double* y) {
    // Create Kokkos views from raw pointers (unmanaged)
    Kokkos::View<const double*, Kokkos::MemoryTraits<Kokkos::Unmanaged>> x_view(x, n);
    Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::Unmanaged>> y_view(y, n);
    
    Kokkos::parallel_for("daxpy", n, KOKKOS_LAMBDA(const int i) {
        y_view(i) += a * x_view(i);
    });
    
    // Ensure completion
    Kokkos::fence();
}

// Kokkos version with managed Views
void daxpy_kokkos_managed(double a, Kokkos::View<double*> x, Kokkos::View<double*> y) {
    const int n = x.extent(0);
    
    Kokkos::parallel_for("daxpy_managed", n, KOKKOS_LAMBDA(const int i) {
        y(i) += a * x(i);
    });
    
    Kokkos::fence();
}

// Kokkos version with team policy for better performance on some architectures
void daxpy_kokkos_team(int n, double a, const double* x, double* y) {
    // Create Kokkos views from raw pointers
    Kokkos::View<const double*, Kokkos::MemoryTraits<Kokkos::Unmanaged>> x_view(x, n);
    Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::Unmanaged>> y_view(y, n);
    
    // Use team policy for better performance on GPU/many-core systems
    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    
    const int league_size = (n + 127) / 128;  // Number of teams
    const int team_size = 128;                // Team size
    
    Kokkos::parallel_for("daxpy_team", 
        team_policy(league_size, team_size),
        KOKKOS_LAMBDA(const member_type& team) {
            const int start = team.league_rank() * team_size;
            const int end = Kokkos::min(start + team_size, n);
            
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, start, end),
                [=](const int i) {
                    y_view(i) += a * x_view(i);
                });
        });
    
    Kokkos::fence();
}

// Demonstration and testing
int main() {
    // Initialize Kokkos
    Kokkos::initialize();
    {
        const int n = 1000000;
        const double alpha = 2.5;
        
        // Initialize vectors
        std::vector<double> x(n);
        std::vector<double> y(n);
        std::vector<double> y_orig(n);
        
        // Fill with test data
        for (int i = 0; i < n; i++) {
            x[i] = static_cast<double>(i + 1);
            y[i] = static_cast<double>(2 * i + 1);
            y_orig[i] = y[i];  // Save original for comparison
        }
        
        std::cout << "Running on Kokkos execution space: " 
                  << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;
        
        // Test basic DAXPY
        auto start = std::chrono::high_resolution_clock::now();
        daxpy(n, alpha, x.data(), 1, y.data(), 1);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Basic DAXPY time: " << duration.count() << " microseconds\n";
        
        // Verify first few results
        std::cout << "\nFirst 5 results (y = " << alpha << "*x + y_orig):\n";
        for (int i = 0; i < 5; i++) {
            double expected = alpha * x[i] + y_orig[i];
            std::cout << "y[" << i << "] = " << std::fixed << std::setprecision(2) 
                      << y[i] << " (expected: " << expected << ")\n";
        }
        
        // Reset y for Kokkos version test
        y = y_orig;
        
        // Test Kokkos version
        start = std::chrono::high_resolution_clock::now();
        daxpy_kokkos(n, alpha, x.data(), y.data());
        end = std::chrono::high_resolution_clock::now();
        
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "\nKokkos DAXPY time: " << duration.count() << " microseconds\n";
        
        // Verify Kokkos results
        bool kokkos_correct = true;
        for (int i = 0; i < 5; i++) {
            double expected = alpha * x[i] + y_orig[i];
            if (std::abs(y[i] - expected) > 1e-10) {
                kokkos_correct = false;
                break;
            }
        }
        std::cout << "Kokkos version correctness: " << (kokkos_correct ? "PASS" : "FAIL") << "\n";
        
        // Reset y for team policy test
        y = y_orig;
        
        // Test Kokkos team policy version
        start = std::chrono::high_resolution_clock::now();
        daxpy_kokkos_team(n, alpha, x.data(), y.data());
        end = std::chrono::high_resolution_clock::now();
        
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Kokkos Team DAXPY time: " << duration.count() << " microseconds\n";
        
        // Test with Kokkos managed Views
        Kokkos::View<double*> x_managed("x_managed", n);
        Kokkos::View<double*> y_managed("y_managed", n);
        
        // Initialize managed views
        Kokkos::parallel_for("init_x", n, KOKKOS_LAMBDA(const int i) {
            x_managed(i) = static_cast<double>(i + 1);
        });
        
        Kokkos::parallel_for("init_y", n, KOKKOS_LAMBDA(const int i) {
            y_managed(i) = static_cast<double>(2 * i + 1);
        });
        
        Kokkos::fence();
        
        start = std::chrono::high_resolution_clock::now();
        daxpy_kokkos_managed(alpha, x_managed, y_managed);
        end = std::chrono::high_resolution_clock::now();
        
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Kokkos Managed DAXPY time: " << duration.count() << " microseconds\n";
        
        // Reset y for STL version test
        y = y_orig;
        
        // Test STL version
        start = std::chrono::high_resolution_clock::now();
        daxpy_stl(alpha, x, y);
        end = std::chrono::high_resolution_clock::now();
        
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "STL DAXPY time: " << duration.count() << " microseconds\n";
        
        // Test template version with float
        std::vector<float> x_f(10), y_f(10);
        for (int i = 0; i < 10; i++) {
            x_f[i] = static_cast<float>(i + 1);
            y_f[i] = static_cast<float>(i * 2);
        }
        
        axpy(10, 1.5f, x_f.data(), 1, y_f.data(), 1);
        
        std::cout << "\nTemplate version with float (first 5 results):\n";
        for (int i = 0; i < 5; i++) {
            std::cout << "y_f[" << i << "] = " << y_f[i] << "\n";
        }
        
    } // End of Kokkos scope
    
    // Finalize Kokkos
    Kokkos::finalize();
    
    return 0;
}
