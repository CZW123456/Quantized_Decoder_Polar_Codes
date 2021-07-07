//
// Created by Zhiwei Cao on 2021/3/17.
//

#ifndef LIBPOLARBD_DMETRIC_H
#define LIBPOLARBD_DMETRIC_H

#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "utils.h"
using namespace std;
namespace py = pybind11;

class DMetricCalculator {
public:
    DMetricCalculator(int N, int K, vector<int> frozen_bits, vector<int> message_bits, vector<int> node_type);
    double calculate(py::array_t<double>& llr);
private:
    int N;
    int K;
    vector<int> frozen_bits;
    vector<int> message_bits;
    std::vector<int> node_type;
};

#endif //LIBPOLARBD_DMETRIC_H
