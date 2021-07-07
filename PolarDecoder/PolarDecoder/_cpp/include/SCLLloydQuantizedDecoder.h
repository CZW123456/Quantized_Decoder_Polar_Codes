//
// Created by Zhiwei Cao on 2020/9/4.
//

#ifndef _LIBPOLARDECODER_SCLLLOYDQUANTIZEDDECODER_H
#define _LIBPOLARDECODER_SCLLLOYDQUANTIZEDDECODER_H

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "utils.h"
namespace py = pybind11;
using namespace std;

#define DOUBLE_INF 1e300

class SCLLloydQuantizedDecoder {
public:
    SCLLloydQuantizedDecoder(int N_, int K_, int _L, vector<int> frozen_bits_, vector<int> message_bits_,
                            vector<vector<double>> &_boundaries_f, vector<vector<double>> &_boundaries_g,
                            vector<vector<double>> &_reconstruction_f, vector<vector<double>> &_reconstruction_g,
                            int _v);
    py::array_t<uint8_t> decode(py::array_t<double>& llr);

private:
    int N;
    int K;
    int L;
    int v;
    vector<int> frozen_bits;
    vector<int> message_bits;
    vector<vector<double>> boundaries_f;
    vector<vector<double>> boundaries_g;
    vector<vector<double>> reconstruction_f;
    vector<vector<double>> reconstruction_g;
};

#endif //_LIBPOLARDECODER_SCLLLOYDQUANTIZEDDECODER_H
