//
// Created by Zhiwei Cao on 2020/3/26.
//

#include "SCLUTDecoder.h"
using namespace std;
namespace py = pybind11;


SCLUT::SCLUT(int N_, int K_, vector<int> frozen_bits_, vector<int> message_bits_, vector<vector<vector<vector<int>>>> lut_f_,
             vector<vector<vector<vector<vector<int>>>>> lut_g_, vector<vector<vector<double>>> virtual_channel_llr_) {
    N = N_;
    K = K_;
    frozen_bits = move(frozen_bits_);
    message_bits = move(message_bits_);
    lut_fs = move(lut_f_);
    lut_gs = move(lut_g_);
    virtual_channel_llrs = move(virtual_channel_llr_);
}

py::array_t<uint8_t> SCLUT::decode(py::array_t<int> &channel_quantized_symbols) {
    int n = (int)log2f((float)N);
    int depth = 0;
    int node = 0;
    bool done = false;
    py::buffer_info buf = channel_quantized_symbols.request();
    vector<ssize_t> size_buf = buf.shape;
    assert(buf.shape[0] == 1);
    std::vector<int> node_state(2 * N - 1);
    std::vector<int> symbols((n + 1) * N);
    std::vector<u_int8_t> ucap((n + 1) * N);
    memcpy(symbols.data(), (int*)channel_quantized_symbols.data(), sizeof(int) * N);
    while (!done){
        if (depth == n){
            if (node == N - 1){
                done = true;
            } else{
                node /= 2;
                --depth;
            }
        } else{
            int node_posi = (1 << depth) + node - 1;
            if (node_state[node_posi] == 0){
                int temp = 1 << (n - depth);
                auto *pa = (int*)symbols.data() + (depth*N + temp*node);
                auto *pb = (int*)symbols.data() + (depth*N + temp*node + temp/2);

                node *= 2;
                depth++;
                temp /= 2;

                // replace f function with LUT

                if (depth < n){
                    auto *psymbols = (int*)symbols.data() + (depth*N + temp*node);
                    for (int i = 0; i < temp; ++i) {
                        *psymbols++ = lut_fs[node_posi][i][pa[i]][pb[i]];
                    }
                } else{
                    if (frozen_bits[node] == 1){
                        ucap[n*N+node] = 0;
                    } else{
                        int symbol = lut_fs[node_posi][0][pa[0]][pb[0]];
                        double llr = virtual_channel_llrs[n - 1][node][symbol];
                        ucap[n*N+node] = (u_int8_t)(llr <= 0);
                    }
                }
                node_state[node_posi] = 1;
            } else if (node_state[node_posi] == 1){
                int temp = 1 << (n - depth);
                auto * pa = (int*)symbols.data() + (depth*N + temp*node);
                auto * pb = (int*)symbols.data() + (depth*N + temp*node + temp/2);

                int ltemp = temp/2;
                int lnode = 2 * node;
                int cdepth = depth + 1;
                auto *pucapl = (u_int8_t*)ucap.data() + (cdepth*N + lnode * ltemp);

                node = 2 * node + 1;
                depth++;
                temp /= 2;

                // replace g function using LUT

                if (depth < n){
                    auto* psymbols = (int*)symbols.data() + (depth*N + node*temp);
                    for (int i = 0; i < temp; ++i) {
                        *psymbols++ = lut_gs[node_posi][i][int(pucapl[i])][pa[i]][pb[i]];
                    }
                } else{
                    if (frozen_bits[node] == 1){
                        ucap[n*N+node] = 0;
                    } else{
                        int symbol = lut_gs[node_posi][0][int(pucapl[0])][pa[0]][pb[0]];
                        double llr = virtual_channel_llrs[n - 1][node][symbol];
                        ucap[n*N+node] = (u_int8_t)(llr <= 0);
                    }
                }
                node_state[node_posi] = 2;
            } else{
                int temp = 1 << (n - depth);
                int ctemp = temp / 2;
                int lnode = 2 * node;
                int rnode = 2 * node + 1;
                int cdepth = depth + 1;
                auto *pucapl = (u_int8_t*)ucap.data() + (cdepth*N + ctemp*lnode);
                auto* pucapr = (u_int8_t*)ucap.data() + (cdepth*N + ctemp*rnode);
                auto* pucap = (u_int8_t*)ucap.data() + (depth*N + temp*node);
                u(pucap, pucapl, pucapr, ctemp);
                node /= 2;
                depth--;
                node_state[node_posi] = 3;
            }
        }
    }
    auto out = py::array_t<uint8_t>(K);
    py::buffer_info buf_result = out.request();
    auto* ptr_result = (uint8_t*)buf_result.ptr;
    for (int i = 0; i < N; ++i) {
        if (frozen_bits[i] == 0){
            *ptr_result++ = ucap[n*N+i];
        }
    }
    return out;
}