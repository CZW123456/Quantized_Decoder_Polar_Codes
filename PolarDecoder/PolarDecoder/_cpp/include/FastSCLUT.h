//
// Created by Zhiwei Cao on 2020/3/27.
//

#ifndef _LIBPOLARDECODER_FASTSCLUT_H
#define _LIBPOLARDECODER_FASTSCLUT_H
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include "utils.h"
using namespace std;
namespace py = pybind11;

class FastSCLUT {
public:
    FastSCLUT(int N_, int K_, vector<int> frozen_bits_, vector<int> message_bits_, vector<int> node_type_, vector<vector<vector<vector<int>>>> lut_f_, vector<vector<vector<vector<vector<int>>>>> lut_g_, vector<vector<vector<double>>> virtual_channel_llr_);
    py::array_t<uint8_t> decode(py::array_t<int>& channel_quantized_symbols);
private:
    int N;
    int K;
    vector<int> frozen_bits;
    vector<int> message_bits;
    std::vector<int> node_type;
    vector<vector<vector<vector<int>>>> lut_fs;
    vector<vector<vector<vector<vector<int>>>>> lut_gs;
    vector<vector<vector<double>>> virtual_channel_llrs;
};
#endif //_LIBPOLARDECODER_FASTSCLUT_H
