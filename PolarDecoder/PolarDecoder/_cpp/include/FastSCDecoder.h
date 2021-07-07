//
// Created by Zhiwei Cao on 2019/12/14.
//

#ifndef POLARDECODER_FASTSCDECODER_H
#define POLARDECODER_FASTSCDECODER_H
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "utils.h"
using namespace std;
namespace py = pybind11;

class FastSC {
public:
    FastSC(int N, int K, vector<int> frozen_bits, vector<int> message_bits, vector<int> node_type);
    py::array_t<uint8_t> decode(py::array_t<double>& llr);
private:
    int N;
    int K;
    vector<int> frozen_bits;
    vector<int> message_bits;
    std::vector<int> node_type;
};
#endif //POLARDECODER_FASTSCDECODER_H
