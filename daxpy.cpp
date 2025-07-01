#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

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

// Demonstration and testing
int main() {
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
    
    // Reset y for STL version test
    y = y_orig;
    
    // Test STL version
    start = std::chrono::high_resolution_clock::now();
    daxpy_stl(alpha, x, y);
    end = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "\nSTL DAXPY time: " << duration.count() << " microseconds\n";
    
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
    
    return 0;
}
