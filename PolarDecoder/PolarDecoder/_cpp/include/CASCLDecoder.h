//
// Created by Zhiwei Cao on 2020/6/6.
//

#ifndef _LIBPOLARDECODER_CA_SCLDECODER_H
#define _LIBPOLARDECODER_CA_SCLDECODER_H
#endif //_LIBPOLARDECODER_CA_SCLDECODER_H

#define DOUBLE_INF 1e300

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <cmath>
#include <vector>
#include "utils.h"

using namespace std;
namespace py = pybind11;

class CASCL {
public:
    CASCL(int N_, int K_, int A_, int L_, vector<int>& frozen_bits_, vector<int>& message_bits_, int crc_n_, vector<int>& crc_p_);
    py::array_t<uint8_t> decode(py::array_t<double> channel_llr);
private:
    vector<int> crc_encoding(vector<int> &info);
    int N;
    int K;
    int L;
    int A;
    vector<int> frozen_bits;
    vector<int> message_bits;
    int crc_n;
    vector<int> crc_p;
};
