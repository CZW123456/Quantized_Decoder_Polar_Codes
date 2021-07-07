//
// Created by Zhiwei Cao on 2019/12/14.
//

#ifndef POLARDECODER_SCLDECODER_H
#define POLARDECODER_SCLDECODER_H

#define DOUBLE_INF 1e300

#include "stdio.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <cmath>
#include <vector>
#include "utils.h"

using namespace std;
namespace py = pybind11;

class SCL {
public:
    SCL(int N_, int K_, int L_, vector<int> frozen_bits_, vector<int> message_bits_);
    py::array_t<uint8_t> decode(py::array_t<double> channel_llr);
private:
    int N;
    int K;
    int L;
    vector<int> frozen_bits;
    vector<int> message_bits;
};
#endif //POLARDECODER_SCLDECODER_H
