#include <iostream>
#include <vector>
#include <stdexcept>
#include <omp.h>

#include "load_heat_matrix.h"

using Matrix = std::vector<std::vector<double>>;

double compute_convo(
    const Matrix& input,
    const Matrix& kernel,
    int row,
    int col
) {
    const int rows = static_cast<int>(input.size());
    const int cols = static_cast<int>(input[0].size());
    const int Nk = static_cast<int>(kernel.size());
    const int pad = Nk / 2;

    double sum = 0.0;

    for (int u = 0; u < Nk; ++u) {
        for (int v = 0; v < Nk; ++v) {
            int r = row + u - pad;
            int c = col + v - pad;

            if (r >= 0 && r < rows &&
                c >= 0 && c < cols) {
                sum += input[r][c] * kernel[u][v];
            }
        }
    }

    return sum;
}

double parallel(
    const Matrix& input,
    const Matrix& kernel,
    Matrix& result
) {
    const int rows = static_cast<int>(input.size());
    const int cols = static_cast<int>(input[0].size());

    double base = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] =
                compute_convo(input, kernel, i, j);
        }
    }

    return omp_get_wtime() - base;
}

double sequential(
    const Matrix& input,
    const Matrix& kernel,
    Matrix& result
) {
    const int rows = static_cast<int>(input.size());
    const int cols = static_cast<int>(input[0].size());

    double base = omp_get_wtime();

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] =
                compute_convo(input, kernel, i, j);
        }
    }

    return omp_get_wtime() - base;
}

int main() {
    try {
        // 1. Load input matrix from CSV.
        Matrix heat = readCSVToMatrix("heat_matrix.csv");

        // 2. Define a 3x3 kernel.
        Matrix kernel = {
            {0.05, 0.1, 0.05},
            {0.1, 0.4, 0.1},
            {0.05, 0.1, 0.05}
        };

        // 3. Validate kernel dimensions.
        if (kernel.empty() ||
            kernel.size() != 3 ||
            kernel[0].size() != 3) {
            throw std::runtime_error(
                "Kernel must be 3x3"
            );
        }

        // 4. Allocate output with the same dimensions.
        const int rows = static_cast<int>(heat.size());
        const int cols = static_cast<int>(heat[0].size());

        Matrix result(rows, std::vector<double>(cols, 0.0));

        // 5. Run parallel convolution.
        double elapsed_para = parallel(heat, kernel, result);

        std::cout << "Input dimensions: "
                  << rows << " x " << cols << '\n';

        std::cout << "Output dimensions: "
                  << result.size() << " x "
                  << result[0].size() << '\n';

        std::cout << "Parallel execution time: "
                  << elapsed_para << " seconds\n";

        // 6. Run sequential convolution.
        double elapsed_seq = sequential(heat, kernel, result);

        std::cout << "Sequential Execution time: "
                  << elapsed_seq << " seconds\n";

        std::cout << "Speedup: "
                  << elapsed_seq / elapsed_para << "x\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
