#include "load_heat_matrix.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

Matrix readCSVToMatrix(const std::string& filename) {
    Matrix matrix;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error(
            "Could not open file: " + filename
        );
    }

    std::string line;

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::vector<double> row;
        std::stringstream ss(line);
        std::string value;

        while (std::getline(ss, value, ',')) {
            try {
                row.push_back(std::stod(value));
            } catch (const std::exception&) {
                throw std::runtime_error(
                    "Invalid numeric value in CSV: " + value
                );
            }
        }

        if (!matrix.empty() &&
            row.size() != matrix.front().size()) {
            throw std::runtime_error(
                "CSV rows have inconsistent column counts"
            );
        }

        matrix.push_back(std::move(row));
    }

    if (matrix.empty() || matrix.front().empty()) {
        throw std::runtime_error("CSV matrix is empty");
    }

    return matrix;
}
