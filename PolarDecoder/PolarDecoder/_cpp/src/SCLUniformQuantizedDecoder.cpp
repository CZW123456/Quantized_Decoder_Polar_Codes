//
// Created by Zhiwei Cao on 2020/6/24.
//

#include "SCLUniformQuantizedDecoder.h"

using namespace std;

template<typename T>
void mink(const std::vector<T>& array, int k, vector<int> &sort_posi, vector<T> &sort_result)
{
    const int array_len(array.size());
    vector<int> array_index(array_len, 0);
    for (int i = 0; i < array_len; ++i){
        array_index[i] = i;
    }
    sort(array_index.begin(), array_index.end(),[&array](int pos1, int pos2) {return (array[pos1] < array[pos2]);});
    memcpy(sort_posi.data(), array_index.data(), sizeof(int)*k);
    for (int i = 0; i < k; ++i) {
        sort_result[i] = array[sort_posi[i]];
    }
}

template<class ForwardIterator>
inline size_t argmin(ForwardIterator first, ForwardIterator last)
{
    return distance(first, min_element(first, last));
}


SCLUniformQuantizedDecoder::SCLUniformQuantizedDecoder(int N_, int K_, int L_, vector<int> frozen_bits_, vector<int> message_bits_,
                                                       vector<double> &_decoder_r_f, vector<double> &_decoder_r_g_, int _v) {
    N = N_;
    K = K_;
    L = L_;
    v = _v;
    frozen_bits = move(frozen_bits_);
    message_bits = move(message_bits_);
    decoder_r_f = _decoder_r_f;
    decoder_r_g = _decoder_r_g_;
}

py::array_t<uint8_t> SCLUniformQuantizedDecoder::decode(py::array_t<double>& llr) {
    py::buffer_info buf = llr.request();
    vector<ssize_t> size_buf = buf.shape;
    assert(buf.shape[0] == 1);
    int n = int(log2((float)N));
    vector<vector<double>> LLR(L);
    vector<vector<u_int8_t>> ucap(L);
    vector<double> PML(L, DOUBLE_INF);
    PML[0] = 0;
    for (int i = 0; i < L; ++i) {
        LLR[i].resize((n + 1) * N);
        memcpy(LLR[i].data(), (double*)buf.ptr, sizeof(double)*N);
        ucap[i].resize((n + 1) * N);
    }
    vector<int> node_state(2 * N - 1);
    int depth = 0;
    int node = 0;
    bool done = false;
    while (!done){
        if (depth == n){
            vector<double> DM(L);
            for (int i = 0; i < L; ++i) {
                DM[i] = LLR[i][n*N+node];
            }
            if (frozen_bits[node] == 1){
                for (int i = 0; i < L; ++i) {
                    ucap[i][n*N+node] = 0;
                    PML[i] += abs(DM[i]) * (float)(DM[i] < 0);
                }
            } else{
                vector<uint8_t> decision(L);
                vector<double> PM2(2*L);
                // path expansion
                memcpy(PM2.data(), PML.data(), sizeof(double)*L);
                for (int i = 0; i < L; ++i) {
                    decision[i] = (uint8_t)(DM[i] < 0);
                    PM2[i + L] = PML[i] +  abs(DM[i]);
                }
                // PM sorting and select K best path
                vector<int> posi(L);
                mink(PM2, L, posi, PML);
                vector<bool> posi1(L, false);
                // determine the decision bits for each path
                vector<uint8_t> tmp_decision = decision;
                // determine the decision bits for each path
                for (int i = 0; i < L; ++i) {
                    if (posi[i] >= L){
                        posi1[i] = true;
                        posi[i] -= L;
                    }
                    tmp_decision[i] = decision[posi[i]];
                }
                decision = tmp_decision;
                for (int i = 0; i < L; ++i) {
                    if (posi1[i]){
                        decision[i] = 1 - decision[i];
                    }
                }
                // rearrange LLR tensor and ucap tensor
                vector<vector<double>> tmp_LLR = LLR;
                vector<vector<uint8_t>> tmp_ucap = ucap;
                // rearrange LLR tensor and ucap tensor
                for (int i = 0; i < L; ++i) {
                    tmp_LLR[i] = LLR[posi[i]];
                    tmp_ucap[i] = ucap[posi[i]];
                    tmp_ucap[i][n*N+node] = decision[i];
                }
                LLR = tmp_LLR;
                ucap = tmp_ucap;
            }
            if (node == N - 1){
                done = true;
            } else{
                node /= 2;
                depth--;
            }
        } else {
            int node_posi = (1 << depth) + node - 1;
            if (node_state[node_posi] == 0){
                int temp = 1 << (n - depth);
                double r_f = decoder_r_f[node_posi];
                double M = double(v/2 - 0.5) * r_f;
                // perform f operation on each SC decoder
                for (int i = 0; i < L; ++i) {
                    auto *pa = (double*)LLR[i].data() + (depth*N + temp*node);
                    auto *pb = (double*)LLR[i].data() + (depth*N + temp*node + temp/2);
                    int ctemp = temp / 2;
                    int lnode = 2 * node;
                    int cdepth = depth + 1;
                    auto *pllr = (double*)LLR[i].data() + (cdepth*N + ctemp*lnode);
                    q_f(pllr, pa, pb, ctemp, r_f, M);
                }
                node *= 2;
                depth++;
                node_state[node_posi] = 1;
            } else if (node_state[node_posi] == 1){
                int temp = 1 << (n - depth);
                double r_g = decoder_r_g[node_posi];
                double M = double(v/2 - 1) * r_g;
                // perform g operation on each SC decoder
                for (int i = 0; i < L; ++i) {
                    auto * pa = (double*)LLR[i].data() + (depth*N + temp*node);
                    auto * pb = (double*)LLR[i].data() + (depth*N + temp*node + temp/2);
                    int ctemp = temp/2;
                    int lnode = 2 * node;
                    int rnode = 2 * node + 1;
                    int cdepth = depth + 1;
                    auto *pucapl = (uint8_t*)ucap[i].data() + (cdepth*N + lnode * ctemp);
                    auto* pllr = (double*)LLR[i].data() + (cdepth*N + rnode*ctemp);
                    q_g(pllr, pa, pb, pucapl, ctemp, r_g, M);
                }
                node = 2 * node + 1;
                depth++;
                node_state[node_posi] = 2;
            } else{
                int temp = 1 << (n - depth);
                for (int i = 0; i < L; ++i) {
                    int ctemp = temp / 2;
                    int lnode = 2 * node;
                    int rnode = 2 * node + 1;
                    int cdepth = depth + 1;
                    auto *pucapl = (u_int8_t*)ucap[i].data() + (cdepth*N + ctemp*lnode);
                    auto* pucapr = (u_int8_t*)ucap[i].data() + (cdepth*N + ctemp*rnode);
                    auto* pucap = (u_int8_t*)ucap[i].data() + (depth*N + temp*node);
                    u(pucap, pucapl, pucapr, ctemp);
                }
                node /= 2;
                depth--;
                node_state[node_posi] = 3;
            }
        }
    }
    int idx = argmin(PML.begin(), PML.end());
    auto out = py::array_t<uint8_t>(K);
    py::buffer_info buf_result = out.request();
    auto* ptr_result = (uint8_t*)buf_result.ptr;
    for (int i = 0; i < N; ++i) {
        if (frozen_bits[i] == 0){
            *ptr_result++ = ucap[idx][n*N+i];
        }
    }
    return out;
}