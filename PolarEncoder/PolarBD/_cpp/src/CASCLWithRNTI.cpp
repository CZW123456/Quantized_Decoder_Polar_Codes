//
// Created by Zhiwei Cao on 2021/3/25.
//

#include "CASCLWithRNTI.h"

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

template<typename T> inline std::vector<int> argsort(const std::vector<T>& array)
{
    const int array_len(array.size());
    std::vector<int> array_index(array_len, 0);
    for (int i = 0; i < array_len; ++i)
        array_index[i] = i;

    std::sort(array_index.begin(), array_index.end(),[&array](int pos1, int pos2) {return (array[pos1] < array[pos2]);});

    return array_index;
}

CASCL::CASCL(int N_, int K_, int A_, int L_, vector<int>& frozen_bits_, vector<int>& message_bits_, int crc_n_, vector<int>& crc_loc) {
    N = N_;
    K = K_;
    A = A_;
    L = L_;
    frozen_bits = move(frozen_bits_);
    message_bits = move(message_bits_);
    crc_n = crc_n_;
    crc_p.resize(crc_n + 1, 0);
    for (int i : crc_loc) {
        crc_p[i] = 1;
    }
}

vector<int> CASCL::crc_encoding(vector<int> &info) {
    int info_length = info.size();
    int times = info_length;
    int n = crc_n + 1;
    vector<int> u(info_length + crc_n, 0);
    memcpy((int*)u.data(), (int*)info.data(), sizeof(int)*info_length);
    for (int i = 0; i < times; ++i) {
        if (u[i] == 1) {
            for (int j = 0; j < n; ++j) {
                u[j + i] = (u[j + i] + crc_p[j]) % 2;
            }
        }
    }
    vector<int> check_code(crc_n);
    memcpy(check_code.data(), u.data() + info_length, sizeof(int)*crc_n);
    return check_code;
}

py::tuple CASCL::decode(py::array_t<double> channel_llr, const py::array_t<int> &RNTI) {
    py::buffer_info buf = channel_llr.request();
    vector<ssize_t> size_buf = buf.shape;
    assert(buf.shape[0] == 1);
    int n = int(log2((float)N));
    int RNTILength = RNTI.shape()[0];
    vector<vector<double>> LLR(L);
    vector<vector<u_int8_t>> ucap(L);
    vector<double> PML(L, 1e30);
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
                // perform f operation on each SC decoder
                for (int i = 0; i < L; ++i) {
                    auto *pa = (double*)LLR[i].data() + (depth*N + temp*node);
                    auto *pb = (double*)LLR[i].data() + (depth*N + temp*node + temp/2);
                    int ctemp = temp / 2;
                    int lnode = 2 * node;
                    int cdepth = depth + 1;
                    auto *pllr = (double*)LLR[i].data() + (cdepth*N + ctemp*lnode);
                    f(pllr, pa, pb, ctemp);
                }
                node *= 2;
                depth++;
                node_state[node_posi] = 1;
            } else if (node_state[node_posi] == 1){
                int temp = 1 << (n - depth);
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
                    g(pllr, pa, pb, pucapl, ctemp);
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
    // CRC check to improve error correction performance
    vector<int> sorted_idx = argsort<double>(PML);
    int winner = sorted_idx[0];
    double PM = PML[0];
    bool isPass = false;
    for (int i = 0; i < L; ++i) {
        vector<int> decoded_info(K);
        vector<int> info(A);
        int cnt = 0;
        for (int k = 0; k < N; ++k) {
            if (frozen_bits[k] == 0) {
                decoded_info[cnt] = (int)ucap[sorted_idx[i]][n * N + k];
                if (cnt < A){
                    info[cnt] = decoded_info[cnt];
                }
                ++cnt;
            }
        }
        vector<int> check_code = crc_encoding(info);
        for (int l = 0; l < RNTILength; ++l) {
            check_code[l + crc_n - RNTILength] = (check_code[l + crc_n - RNTILength] + RNTI.data()[l]) % 2;
        }
        bool is_pass = true;
        for (int j = 0; j < crc_n; ++j) {
            if (check_code[j] != decoded_info[A + j]) {
                is_pass = false;
                break;
            }
        }
        if (is_pass) {
            winner = sorted_idx[i];
            PM = PML[i];
            isPass = true;
            break;
        }
    }
    auto out = py::array_t<uint8_t>(A);
    py::buffer_info buf_result = out.request();
    auto* ptr_result = (uint8_t*)buf_result.ptr;
    int cnt = 0;
    for (int i = 0; i < N; ++i) {
        if (frozen_bits[i] == 0){
            *ptr_result++ = ucap[winner][n*N+i];
            cnt++;
            if (cnt == A){
                break;
            }
        }
    }
    return py::make_tuple(out, PM, isPass);
}