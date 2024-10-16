#include <iostream>
#include <chrono>
#include <random>
#include <Eigen/Dense>

const int MATRIX_SIZE = 1000;

int main() {
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Create matrices
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(MATRIX_SIZE, MATRIX_SIZE);
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(MATRIX_SIZE, MATRIX_SIZE);

    // Fill matrices with random values
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            A(i, j) = dis(gen);
            B(i, j) = dis(gen);
        }
    }

    // Perform matrix multiplication and measure time
    std::cout << "Testing C++ (Eigen) matrix multiplication" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    Eigen::MatrixXd C = A * B;
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "C++ (Eigen) matrix multiplication time: " << diff.count() << " seconds" << std::endl;

    return 0;
}
