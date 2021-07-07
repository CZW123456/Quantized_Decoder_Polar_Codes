//
// Created by Zhiwei Cao on 2020/6/8.
//

#include "CAFastSCLLUTDecoder.h"
#include <vector>
using namespace std;
template<typename T> inline vector<int> argsort(const std::vector<T>& array)
{
    const int array_len(array.size());
    std::vector<int> array_index(array_len, 0);
    for (int i = 0; i < array_len; ++i)
        array_index[i] = i;

    std::sort(array_index.begin(), array_index.end(),[&array](int pos1, int pos2) {return (array[pos1] < array[pos2]);});

    return array_index;
}

template<typename T>
void mink(const std::vector<T>& array, int k, vector<int> &sort_posi, vector<T> &sort_result)
{
    const int array_len(array.size());
    vector<int> array_index(array_len, 0);
    for (int i = 0; i < array_len; ++i){
        array_index[i] = i;
    }
    sort(array_index.begin(), array_index.end(),[&array](int pos1, int pos2) {
        return (array[pos1] < array[pos2]);
    });
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

CAFastSCLLUT::CAFastSCLLUT(int N_, int K_, int A_, int L_, vector<int> frozen_bits_, vector<int> message_bits_,
                           vector<int> node_type_, vector<vector<vector<vector<int>>>> lut_f_,
                           vector<vector<vector<vector<vector<int>>>>> lut_g_,
                           vector<vector<vector<double>>> virtual_channel_llr_) {
    N = N_;
    K = K_;
    A = A_;
    L = L_;
    frozen_bits = move(frozen_bits_);
    message_bits = move(message_bits_);
    node_type = move(node_type_);
    lut_fs = move(lut_f_);
    lut_gs = move(lut_g_);
    virtual_channel_llrs = move(virtual_channel_llr_);
}


py::array_t<uint8_t> CAFastSCLLUT::decode(py::array_t<int> &channel_quantized_symbols) {
    py::buffer_info buf = channel_quantized_symbols.request();
    vector <ssize_t> size_buf = buf.shape;
    assert(buf.shape[0] == 1);
    int n = int(log2((float) N));
    vector <vector<int>> symbols(L);
    vector <vector<u_int8_t>> ucap(L);
    vector<double> PML(L, DOUBLE_INF);
    PML[0] = 0;
    for (int i = 0; i < L; ++i) {
        symbols[i].resize((n + 1) * N);
        memcpy(symbols[i].data(), (int *) buf.ptr, sizeof(int) * N);
        ucap[i].resize((n + 1) * N);
    }
    vector<int> node_state(2 * N - 1);
    int depth = 0;
    int node = 0;
    bool done = false;
    while (!done) {
        if (depth == n) {
            node /= 2;
            --depth;
        } else {
            int node_posi = (1 << depth) + node - 1;
            if (node_state[node_posi] == 0) {
                // R0 node
                if (node_type[node_posi] == 0) {
                    int temp = 1 << (n - depth);
                    for (int i = 0; i < L; ++i) {
                        memset(ucap[i].data() + (depth * N + temp * node), 0, sizeof(uint8_t) * temp);
                        for (int j = 0; j < temp; ++j) {
                            int symbol = symbols[i][depth * N + temp * node + j];
                            double l = virtual_channel_llrs[depth - 1][temp * node + j][symbol];
                            PML[i] += (float) (l < 0) * abs(l);
                        }
                    }
                    node /= 2;
                    depth--;
                    continue;
                }

                // R1 node
                if (node_type[node_posi] == 1) {
                    int temp = 1 << (n - depth);
                    int max_depth = min(L - 1, temp);
                    vector <vector<uint8_t>> decision(L);
                    vector <vector<double>> abs_llr(L);
                    vector <vector<int>> sorted_absllr_idx(L);
                    // compute |LLR| and sort them in ascending order for each SC decoder. compute hard decision result
                    // for each SC decoder by the way
                    for (int i = 0; i < L; ++i) {
                        decision[i].resize(temp);
                        abs_llr[i].resize(temp);
                        for (int j = 0; j < temp; ++j) {
                            int symbol = symbols[i][depth * N + temp * node + j];
                            double l = virtual_channel_llrs[depth - 1][temp * node + j][symbol];
                            decision[i][j] = (uint8_t)(l < 0);
                            abs_llr[i][j] = abs(l);
                        }
                        sorted_absllr_idx[i] = argsort(abs_llr[i]);
                    }
                    // tree search
                    for (int layer = 0; layer < max_depth; ++layer) {
                        vector<double> l_PML = PML;
                        vector<double> r_PML = PML;
                        for (int i = 0; i < L; ++i) {
                            r_PML[i] += abs_llr[i][sorted_absllr_idx[i][layer]];
                        }
                        vector<int> posi(L);
                        vector<double> PML2(2 * L);
                        memcpy(PML2.data(), l_PML.data(), sizeof(double) * L);
                        memcpy(PML2.data() + L, r_PML.data(), sizeof(double) * L);
                        mink(PML2, L, posi, PML);
                        vector<bool> posi1(L, false);
                        for (int i = 0; i < L; ++i) {
                            if (posi[i] >= L) {
                                posi1[i] = true;
                                posi[i] -= L;
                            }
                        }
                        vector <vector<u_int8_t>> tmp_decision = decision;
                        vector <vector<int>> tmp_symbols = symbols;
                        vector <vector<uint8_t>> tmp_ucap = ucap;
                        vector <vector<int>> tmp_sorted_absllr_idx = sorted_absllr_idx;
                        vector <vector<double>> tmp_abs_llr = abs_llr;
                        for (int i = 0; i < L; ++i) {
                            tmp_decision[i] = decision[posi[i]];
                            if (posi1[i]) {
                                tmp_decision[i][sorted_absllr_idx[i][layer]] =
                                        1 - tmp_decision[i][sorted_absllr_idx[i][layer]];
                            }
                            tmp_symbols[i] = symbols[posi[i]];
                            tmp_ucap[i] = ucap[posi[i]];
                            tmp_abs_llr[i] = abs_llr[posi[i]];
                            tmp_sorted_absllr_idx[i] = sorted_absllr_idx[posi[i]];
                        }
                        decision = tmp_decision;
                        symbols = tmp_symbols;
                        ucap = tmp_ucap;
                        abs_llr = tmp_abs_llr;
                        sorted_absllr_idx = tmp_sorted_absllr_idx;
                    }
                    // copy result
                    for (int i = 0; i < L; ++i) {
                        memcpy(ucap[i].data() + (depth * N + temp * node), decision[i].data(), sizeof(uint8_t) * temp);
                    }
                    // return to parent node
                    node /= 2;
                    depth--;
                    continue;
                }

                // REP node
                if (node_type[node_posi] == 2) {
                    int temp = 1 << (n - depth);
                    vector <vector<double>> abs_llr(L);
                    vector<double> PML2(2 * L);
                    memcpy(PML2.data(), PML.data(), sizeof(double) * L);
                    memcpy(PML2.data() + L, PML.data(), sizeof(double) * L);
                    for (int i = 0; i < L; ++i) {
                        abs_llr[i].resize(temp);
                        for (int j = 0; j < temp; ++j) {
                            int symbol = symbols[i][depth * N + temp * node + j];
                            double l = virtual_channel_llrs[depth - 1][temp * node + j][symbol];
                            abs_llr[i][j] = abs(l);
                            PML2[i] += (double) (l < 0) * abs(l);
                            PML2[i + L] += (double) (l >= 0) * abs_llr[i][j];
                        }
                    }
                    // sort 2L valid codeword and obtain L best codeword, each codeword is either all 0 or all 1
                    vector<int> posi(L);
                    mink(PML2, L, posi, PML);
                    // rearrange tensors
                    vector <vector<uint8_t>> decision(L);
                    vector<bool> posi1(L);
                    vector <vector<int>> tmp_symbols = symbols;
                    vector <vector<uint8_t>> tmp_ucap = ucap;
                    for (int i = 0; i < L; ++i) {
                        decision[i].resize(temp, 0);
                        if (posi[i] >= L) {
                            posi1[i] = true;
                            posi[i] -= L;
                            memset(decision[i].data(), 1, sizeof(uint8_t) * temp);
                        } else {
                            memset(decision[i].data(), 0, sizeof(uint8_t) * temp);
                        }
                        tmp_symbols[i] = symbols[posi[i]];
                        tmp_ucap[i] = ucap[posi[i]];
                    }
                    ucap = tmp_ucap;
                    symbols = tmp_symbols;
                    for (int i = 0; i < L; ++i) {
                        memcpy(ucap[i].data() + (depth * N + temp * node), decision[i].data(), sizeof(uint8_t) * temp);
                    }
                    node /= 2;
                    depth--;
                    continue;
                }

                //TODO: Add SPC node

                // perform f operation on each SC decoder
                int temp = 1 << (n - depth);
                int ctemp = temp / 2;
                int lnode = 2 * node;
                int cdepth = depth + 1;

                if (cdepth < n) {
                    for (int i = 0; i < L; ++i) {
                        auto *pa = symbols[i].data() + (depth * N + temp * node);
                        auto *pb = symbols[i].data() + (depth * N + temp * node + ctemp);
                        for (int j = 0; j < ctemp; ++j) {
                            symbols[i][cdepth * N + ctemp * lnode + j] = lut_fs[node_posi][j][pa[j]][pb[j]];
                        }
                    }
                } else {
                    vector<double> DM(L);
                    for (int i = 0; i < L; ++i) {
                        int y = symbols[i][depth * N + temp * node];
                        int x = symbols[i][depth * N + temp * node + ctemp];
                        int symbol = lut_fs[node_posi][0][y][x];
                        double llr = virtual_channel_llrs[n - 1][lnode][symbol];
                        DM[i] = llr;
                    }
                    if (frozen_bits[lnode] == 1) {
                        for (int i = 0; i < L; ++i) {
                            ucap[i][n * N + lnode] = 0;
                            PML[i] += abs(DM[i]) * (double) (DM[i] < 0);
                        }
                    } else {
                        vector <uint8_t> decision(L);
                        vector<double> PM2(2 * L);
                        // path expansion
                        memcpy(PM2.data(), PML.data(), sizeof(double) * L);
                        for (int i = 0; i < L; ++i) {
                            decision[i] = (uint8_t)(DM[i] < 0);
                            PM2[i + L] = PML[i] + abs(DM[i]);
                        }
                        // PM sorting and select K best path
                        vector<int> posi(L);
                        mink(PM2, L, posi, PML);
                        vector<bool> posi1(L, false);
                        // determine the decision bits for each path
                        vector <uint8_t> tmp_decision = decision;
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
                        vector <vector<int>> tmp_symbols = symbols;
                        vector <vector<uint8_t>> tmp_ucap = ucap;
                        // rearrange LLR tensor and ucap tensor
                        for (int i = 0; i < L; ++i) {
                            tmp_symbols[i] = symbols[posi[i]];
                            tmp_ucap[i] = ucap[posi[i]];
                            tmp_ucap[i][n * N + lnode] = decision[i];
                        }
                        symbols = tmp_symbols;
                        ucap = tmp_ucap;
                    }
                }
                node *= 2;
                depth++;
                node_state[node_posi] = 1;
            } else if (node_state[node_posi] == 1) {
                int temp = 1 << (n - depth);
                int ctemp = temp / 2;
                int lnode = 2 * node;
                int rnode = 2 * node + 1;
                int cdepth = depth + 1;
                // perform g operation on each SC decoder
                if (cdepth < n) {
                    for (int i = 0; i < L; ++i) {
                        auto *pa = symbols[i].data() + (depth * N + temp * node);
                        auto *pb = symbols[i].data() + (depth * N + temp * node + ctemp);
                        auto *pucapl = ucap[i].data() + (cdepth * N + lnode * ctemp);
                        for (int j = 0; j < ctemp; ++j) {
                            symbols[i][cdepth * N + ctemp * rnode + j] = lut_gs[node_posi][j][int(
                                    pucapl[j])][pa[j]][pb[j]];
                        }
                    }
                } else {
                    vector<double> DM(L);
                    for (int i = 0; i < L; ++i) {
                        int ucapl = ucap[i][cdepth * N + lnode * ctemp];
                        int y = symbols[i][depth * N + temp * node];
                        int x = symbols[i][depth * N + temp * node + ctemp];
                        int symbol = lut_gs[node_posi][0][ucapl][y][x];
                        double llr = virtual_channel_llrs[n - 1][rnode][symbol];
                        DM[i] = llr;
                    }
                    if (frozen_bits[rnode] == 1) {
                        for (int i = 0; i < L; ++i) {
                            ucap[i][n * N + rnode] = 0;
                            PML[i] += abs(DM[i]) * (double) (DM[i] < 0);
                        }
                    } else {
                        vector <uint8_t> decision(L);
                        vector<double> PM2(2 * L);
                        // path expansion
                        memcpy(PM2.data(), PML.data(), sizeof(double) * L);
                        for (int i = 0; i < L; ++i) {
                            decision[i] = (uint8_t)(DM[i] < 0);
                            PM2[i + L] = PML[i] + abs(DM[i]);
                        }
                        // PM sorting and select L best path
                        vector<int> posi(L);
                        mink(PM2, L, posi, PML);
                        vector<bool> posi1(L, false);
                        // determine the decision bits for each path
                        vector <uint8_t> tmp_decision = decision;
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
                        vector <vector<int>> tmp_symbols = symbols;
                        vector <vector<uint8_t>> tmp_ucap = ucap;
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
            } else {
                int temp = 1 << (n - depth);
                if (node == 0 and depth == 0) {
                    done = true;
                }
                for (int i = 0; i < L; ++i) {
                    int ctemp = temp / 2;
                    int lnode = 2 * node;
                    int rnode = 2 * node + 1;
                    int cdepth = depth + 1;
                    auto *pucapl = (u_int8_t *) ucap[i].data() + (cdepth * N + ctemp * lnode);
                    auto *pucapr = (u_int8_t *) ucap[i].data() + (cdepth * N + ctemp * rnode);
                    auto *pucap = (u_int8_t *) ucap[i].data() + (depth * N + temp * node);
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
    for (int i = 0; i < L; ++i) {
        vector<uint8_t> x(N);
        memcpy(x.data(), ucap[sorted_idx[i]].data(), sizeof(uint8_t)*N);
        int m = 1;
        for (int d = n - 1; d >= 0; --d) {
            for (int i = 0; i < N; i+=2*m) {
                for (int j = 0; j < m; ++j) {
                    x[i+j] = x[i+j] ^ x[i+m+j];
                }
            }
            m *= 2;
        }
        vector<uint8_t> decoded_info(K);
        int cnt = 0;
        for (int k = 0; k < N; ++k) {
            if (frozen_bits[k] == 0){
                decoded_info[cnt] = x[k];
                ++cnt;
            }
        }
        vector<u_int8_t> actual_info(A);
        memcpy(actual_info.data(), decoded_info.data(), sizeof(uint8_t)*A);
        vector<uint8_t> check_code = crc_checker.encoding(actual_info);
        bool is_pass = true;
        for (int j = 0; j < K - A; ++j) {
            if (check_code[j] != decoded_info[A + j]){
                is_pass = false;
                break;
            }
        }
        if (is_pass){
            winner = sorted_idx[i];
            break;
        }
    }
    auto out = py::array_t<uint8_t>(A);
    py::buffer_info buf_result = out.request();
    auto* ptr_result = (uint8_t*)buf_result.ptr;
    int cnt = 0;
    vector<uint8_t> x(N);
    memcpy(x.data(), ucap[winner].data(), sizeof(uint8_t)*N);
    int m = 1;
    for (int d = n - 1; d >= 0; --d) {
        for (int i = 0; i < N; i+=2*m) {
            for (int j = 0; j < m; ++j) {
                x[i+j] = x[i+j] ^ x[i+m+j];
            }
        }
        m *= 2;
    }
    for (int i = 0; i < N; ++i) {
        if (frozen_bits[i] == 0){
            *ptr_result++ = x[i];
            cnt++;
            if (cnt == A){
                break;
            }
        }
    }
    return out;
}