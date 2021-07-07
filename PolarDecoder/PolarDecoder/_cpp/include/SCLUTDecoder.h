//
// Created by Zhiwei Cao on 2020/3/26.
//

#ifndef _LIBPOLARDECODER_SCLUTDECODER_H
#define _LIBPOLARDECODER_SCLUTDECODER_H
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "utils.h"
namespace py = pybind11;
using namespace std;

class SCLUT {
public:
    SCLUT(int N_, int K_, vector<int> frozen_bits_, vector<int> message_bits_, vector<vector<vector<vector<int>>>> lut_f_, vector<vector<vector<vector<vector<int>>>>> lut_g_, vector<vector<vector<double>>> virtual_channel_llr_);
    py::array_t<uint8_t> decode(py::array_t<int>& symbols);
private:
    int N;
    int K;
    vector<int> frozen_bits;
    vector<int> message_bits;
    vector<vector<vector<vector<int>>>> lut_fs;
    vector<vector<vector<vector<vector<int>>>>> lut_gs;
    vector<vector<vector<double>>> virtual_channel_llrs;
};
#endif //_LIBPOLARDECODER_SCLUTDECODER_H
