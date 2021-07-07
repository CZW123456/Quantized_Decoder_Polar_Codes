//
// Created by Zhiwei Cao on 2020/6/8.
//
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include "utils.h"

namespace py = pybind11;
using namespace std;

#define DOUBLE_INF 1.0/0/0

class CAFastSCLLUT {
public:
    CAFastSCLLUT(int N_, int K_, int A_, int L_, vector<int> frozen_bits_, vector<int> message_bits_, vector<int> node_type_, vector<vector<vector<vector<int>>>> lut_f_, vector<vector<vector<vector<vector<int>>>>> lut_g_, vector<vector<vector<double>>> virtual_channel_llr_);
    py::array_t<u_int8_t> decode(py::array_t<int> &channel_quantized_symbols);
private:
    int N;
    int K;
    int A;
    int L;
    vector<int> frozen_bits;
    vector<int> message_bits;
    vector<int> node_type;
    vector<vector<vector<vector<int>>>> lut_fs;
    vector<vector<vector<vector<vector<int>>>>> lut_gs;
    vector<vector<vector<double>>> virtual_channel_llrs;
    vector<int> loc = {24, 23, 21, 20, 17, 15, 13, 12, 8, 4, 2, 1, 0};
    CRC crc_checker = CRC(24, loc);
};
