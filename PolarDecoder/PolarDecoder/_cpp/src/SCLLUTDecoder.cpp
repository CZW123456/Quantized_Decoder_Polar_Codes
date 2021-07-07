//
// Created by Zhiwei Cao on 2020/3/28.
//
#include "SCLLUTDecoder.h"



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



SCLLUT::SCLLUT(int N_, int K_, int L_, vector<int> frozen_bits_, vector<int> message_bits_,
               vector<vector<vector<vector<int>>>> lut_f_, vector<vector<vector<vector<vector<int>>>>> lut_g_,
               vector<vector<vector<double>>> virtual_channel_llr_) {
    N = N_;
    K = K_;
    L = L_;
    frozen_bits = move(frozen_bits_);
    message_bits = move(message_bits_);
    lut_fs = move(lut_f_);
    lut_gs = move(lut_g_);
    virtual_channel_llrs = move(virtual_channel_llr_);
}


py::array_t<uint8_t> SCLLUT::decode(py::array_t<int> &channel_quantized_symbols) {
    py::buffer_info buf = channel_quantized_symbols.request();
    vector<ssize_t> size_buf = buf.shape;
    assert(buf.shape[0] == 1);
    int n = int(log2((float)N));
    vector<vector<int>> symbols(L);
    vector<vector<u_int8_t>> ucap(L);
    vector<double> PML(L, DOUBLE_INF);
    PML[0] = 0;
    for (int i = 0; i < L; ++i) {
        symbols[i].resize((n + 1) * N);
        memcpy(symbols[i].data(), (int*)buf.ptr, sizeof(int)*N);
        ucap[i].resize((n + 1) * N);
    }
    vector<int> node_state(2 * N - 1);
    int depth = 0;
    int node = 0;
    bool done = false;
    while (!done){
        if (depth == n){
            if (node == N - 1){
                done = true;
            } else{
                node /= 2;
                --depth;
            }
        } else {
            int node_posi = (1 << depth) + node - 1;

            if (node_state[node_posi] == 0){
                // perform f operation on each SC decoder
                int temp = 1 << (n - depth);
                int ctemp = temp / 2;
                int lnode = 2 * node;
                int cdepth = depth + 1;

                if (cdepth < n){
                    for (int i = 0; i < L; ++i) {
                        auto *pa = symbols[i].data() + (depth*N + temp*node);
                        auto *pb = symbols[i].data() + (depth*N + temp*node + ctemp);
                        for (int j = 0; j < ctemp; ++j) {
                            symbols[i][cdepth*N + ctemp*lnode + j] = lut_fs[node_posi][j][pa[j]][pb[j]];
                        }
                    }
                } else {
                    vector<double> DM(L);
                    for (int i = 0; i < L; ++i) {
                        int y = symbols[i][depth*N + temp*node];
                        int x = symbols[i][depth*N + temp*node + ctemp];
                        int symbol = lut_fs[node_posi][0][y][x];
                        double llr = virtual_channel_llrs[n - 1][lnode][symbol];
                        DM[i] = llr;
                    }
                    if (frozen_bits[lnode] == 1){
                        for (int i = 0; i < L; ++i) {
                            ucap[i][n*N+lnode] = 0;
                            PML[i] += abs(DM[i]) * (double)(DM[i] < 0);
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
                        vector<vector<int>> tmp_symbols = symbols;
                        vector<vector<uint8_t>> tmp_ucap = ucap;
                        // rearrange LLR tensor and ucap tensor
                        for (int i = 0; i < L; ++i) {
                            tmp_symbols[i] = symbols[posi[i]];
                            tmp_ucap[i] = ucap[posi[i]];
                            tmp_ucap[i][n*N+lnode] = decision[i];
                        }
                        symbols = tmp_symbols;
                        ucap = tmp_ucap;
                    }
                }
                node *= 2;
                depth++;
                node_state[node_posi] = 1;
            } else if (node_state[node_posi] == 1){
                int temp = 1 << (n - depth);
                int ctemp = temp/2;
                int lnode = 2 * node;
                int rnode = 2 * node + 1;
                int cdepth = depth + 1;
                // perform g operation on each SC decoder
                if (cdepth < n){
                    for (int i = 0; i < L; ++i) {
                        auto * pa = symbols[i].data() + (depth*N + temp*node);
                        auto * pb = symbols[i].data() + (depth*N + temp*node + ctemp);
                        auto *pucapl = ucap[i].data() + (cdepth*N + lnode * ctemp);
                        for (int j = 0; j < ctemp; ++j) {
                            symbols[i][cdepth*N + ctemp*rnode + j] = lut_gs[node_posi][j][int(pucapl[j])][pa[j]][pb[j]];
                        }
                    }
                } else {
                    vector<double> DM(L);
                    for (int i = 0; i < L; ++i) {
                        int ucapl = ucap[i][cdepth*N + lnode * ctemp];
                        int y = symbols[i][depth*N+temp*node];
                        int x = symbols[i][depth*N+temp*node+ctemp];
                        int symbol = lut_gs[node_posi][0][ucapl][y][x];
                        double llr = virtual_channel_llrs[n - 1][rnode][symbol];
                        DM[i] = llr;
                    }
                    if (frozen_bits[rnode] == 1) {
                        for (int i = 0; i < L; ++i) {
                            ucap[i][n * N + rnode] = 0;
                            PML[i] += abs(DM[i]) * (double)(DM[i] < 0);
                        }
                    } else {
                        vector<uint8_t> decision(L);
                        vector<double> PM2(2 * L);
                        // path expansion
                        memcpy(PM2.data(), PML.data(), sizeof(double) * L);
                        for (int i = 0; i < L; ++i) {
                            decision[i] = (uint8_t) (DM[i] < 0);
                            PM2[i + L] = PML[i] + abs(DM[i]);
                        }
                        // PM sorting and select L best path
                        vector<int> posi(L);
                        mink(PM2, L, posi, PML);
                        vector<bool> posi1(L, false);
                        // determine the decision bits for each path
                        vector<uint8_t> tmp_decision = decision;
                        // determine the decision bits for each path
                        for (int i = 0; i < L; ++i) {
                            if (posi[i] >= L) {
                                posi1[i] = true;
                                posi[i] -= L;
                            }
                            tmp_decision[i] = decision[posi[i]];
                        }
                        decision = tmp_decision;
                        for (int i = 0; i < L; ++i) {
                            if (posi1[i]) {
                                decision[i] = 1 - decision[i];
                            }
                        }
                        // rearrange LLR tensor and ucap tensor
                        vector<vector<int>> tmp_symbols = symbols;
                        vector<vector<uint8_t>> tmp_ucap = ucap;
                        // rearrange LLR tensor and ucap tensor
                        for (int i = 0; i < L; ++i) {
                            tmp_symbols[i] = symbols[posi[i]];
                            tmp_ucap[i] = ucap[posi[i]];
                            tmp_ucap[i][n * N + rnode] = decision[i];
                        }
                        symbols = tmp_symbols;
                        ucap = tmp_ucap;
                    }
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