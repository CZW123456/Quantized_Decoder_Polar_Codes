//
// Created by Zhiwei Cao on 2019/12/14.
//
#include "SCDecoder.h"
using namespace std;

SC::SC(int N_, int K_, vector<int> frozen_bits_, vector<int> message_bits_) {
    N = N_;
    K = K_;
    frozen_bits = move(frozen_bits_);
    message_bits = move(message_bits_);
}

py::array_t<uint8_t> SC::decode(py::array_t<double>& llr) {
    int n = (int)log2f((float)N);
    int depth = 0;
    int node = 0;
    bool done = false;
    py::buffer_info buf = llr.request();
    vector<ssize_t> size_buf = buf.shape;
    assert(buf.shape[0] == 1);
    std::vector<int> node_state(2 * N - 1);
    std::vector<double> alpha((n + 1) * N);
    std::vector<u_int8_t> ucap((n + 1) * N);
    memcpy(alpha.data(), (double*)llr.data(), sizeof(double) * N);
    while (!done){
        if (depth == n){
            if (frozen_bits[node] == 1){
                ucap[n*N+node] = 0;
            } else{
                ucap[n*N+node] = (uint8_t)(alpha[n*N+node] <= 0);
            }
            if (node == N - 1){
                done = true;
            } else{
                node /= 2;
                depth--;
            }
        } else{
            int node_posi = (1 << depth) + node - 1;
            if (node_state[node_posi] == 0){
                int temp = 1 << (n - depth);
                auto *pa = (double*)alpha.data() + (depth*N + temp*node);
                auto *pb = (double*)alpha.data() + (depth*N + temp*node + temp/2);
                node *= 2;
                depth++;
                temp /= 2;
                auto *pllr = (double*)alpha.data() + (depth*N + temp*node);
                f(pllr, pa, pb, temp);
                node_state[node_posi] = 1;
            } else if (node_state[node_posi] == 1){
                int temp = 1 << (n - depth);
                auto * pa = (double*)alpha.data() + (depth*N + temp*node);
                auto * pb = (double*)alpha.data() + (depth*N + temp*node + temp/2);
                int ltemp = temp/2;
                int lnode = 2 * node;
                int cdepth = depth + 1;
                auto *pucapl = (u_int8_t*)ucap.data() + (cdepth*N + lnode * ltemp);
                node = 2 * node + 1;
                depth++;
                temp /= 2;
                auto* pllr = (double*)alpha.data() + (depth*N + node*temp);
                g(pllr, pa, pb, pucapl, temp);
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
        if (frozen_bits[i] == 0) {
            *ptr_result++ = ucap[n * N + i];
        }
    }
    return out;
}