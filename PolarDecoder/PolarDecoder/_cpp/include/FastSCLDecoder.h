//
// Created by Zhiwei Cao on 2019/12/14.
//

#ifndef POLARDECODER_FASTSCLDECODER_H
#define POLARDECODER_FASTSCLDECODER_H
#define DOUBLE_INF 1.0/0.0

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <cmath>
#include <vector>
#include "utils.h"

using namespace std;
namespace py = pybind11;

class FastSCL {
public:
    FastSCL(int N_, int K_, int L_, vector<int> frozen_bits_, vector<int> message_bits_, vector<int> node_type_);
    py::array_t<u_int8_t> decode(py::array_t<double> channel_llr);
private:
    int N;
    int K;
    int L;
    vector<int> frozen_bits;
    vector<int> message_bits;
    vector<int> node_type;
};
#endif //POLARDECODER_FASTSCLDECODER_H
