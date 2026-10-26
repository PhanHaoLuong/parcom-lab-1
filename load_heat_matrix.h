
#ifndef LOAD_HEAT_MATRIX_H
#define LOAD_HEAT_MATRIX_H

#include <string>
#include <vector>

using Matrix = std::vector<std::vector<double>>;

Matrix readCSVToMatrix(const std::string& filename);

#endif
