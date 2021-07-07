//
// Created by Zhiwei Cao on 2019/12/14.
//

#ifndef POLARDECODER_SCDECODER_H
#define POLARDECODER_SCDECODER_H
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "utils.h"

namespace py = pybind11;
using namespace std;

class SC {
public:
    SC(int N_, int K_, vector<int> frozen_bits_, vector<int> message_bits_);
    py::array_t<uint8_t> decode(py::array_t<double>& llr);

private:
    int N;
    int K;
    vector<int> frozen_bits;
    vector<int> message_bits;
};
#endif //POLARDECODER_SCDECODER_H
