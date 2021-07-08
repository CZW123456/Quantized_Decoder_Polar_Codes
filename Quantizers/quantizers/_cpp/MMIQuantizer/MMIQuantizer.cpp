//
// Created by Zhiwei Cao on 2020/3/11.
//
#include "MMIQuantizer.h"
#include <pybind11/stl.h>
#include <algorithm>

template<typename T> inline std::vector<int> argsort(const std::vector<T>& array)
{
    const int array_len(array.size());
    std::vector<int> array_index(array_len, 0);
    for (int i = 0; i < array_len; ++i)
        array_index[i] = i;

    std::sort(array_index.begin(), array_index.end(),[&array](int pos1, int pos2) {return (array[pos1] < array[pos2]);});

    return array_index;
}

template<class ForwardIterator>
inline size_t argmax(ForwardIterator first, ForwardIterator last)
{
    return std::distance(first, std::max_element(first, last));
}


MMIQuantizer::MMIQuantizer(double p1, double p_minus_1) {
    m_p1 = p1;
    m_p_minus_1 = p_minus_1;
}

Mat MMIQuantizer::precompute_partial_entropy_table(int M, int K) {
    Mat table(M, M+1, CV_64F);
    table.setTo(0);
    auto *p_pyx1 = pyx_1.ptr<double >(0);
    auto *p_p_minus1 = pyx_minus1.ptr<double >(0);
    for (int a_prime = 0; a_prime < M; ++a_prime) {
        int max_a = min(a_prime + M - K + 1, M);
        for (int a = a_prime + 1; a < max_a + 1; ++a){
            int len = a - a_prime;
            Mat tmp_pyx1(1, len, CV_64F), tmp_pyx_minus1(1, len, CV_64F);
            memcpy(tmp_pyx1.ptr<double >(0), p_pyx1 + a_prime, len * sizeof(double_t));
            memcpy(tmp_pyx_minus1.ptr<double_t >(0), p_p_minus1 + a_prime, len * sizeof(double_t));
            table.at<double_t >(a_prime, a) = compute_partial_entropy(tmp_pyx1, tmp_pyx_minus1);
        }
    }
    return table;
}

inline double MMIQuantizer::compute_partial_entropy(Mat &pyx, Mat &pyx_minus_1) {
    double sum_conditional_probability1 = sum(pyx)[0];
    double sum_conditional_probability2 = sum(pyx_minus_1)[0];

    double p_nominator = m_p1 * sum_conditional_probability1 + m_p_minus_1 * sum_conditional_probability2;

    double p1 = sum_conditional_probability1 / p_nominator;
    double p2 = sum_conditional_probability2 / p_nominator;

    Mat tmp1 = Mat::zeros(pyx.size(), CV_64F);
    Mat tmp2 = Mat::zeros(pyx_minus_1.size(), CV_64F);
    if (p1 > 1E-9){
        tmp1 = pyx * log2f(p1);
    }
    if (p2 > 1E-9){
        tmp2 = pyx_minus_1 * log2f(p2);
    }
    double conditional_entropy = m_p1 * sum(tmp1)[0] + m_p_minus_1 * sum(tmp2)[0];

    return conditional_entropy;
}

py::tuple MMIQuantizer::find_opt_quantizer(py::array_t<double> &joint_prob_ndarrary, int K) {
    py::buffer_info buf = joint_prob_ndarrary.request();
    vector<ssize_t> size_buf = buf.shape;
    Mat joint_prob(size_buf[0], size_buf[1], CV_64F, (double*)buf.ptr);
    int M = joint_prob.cols;
    CV_Assert(M >= K);
    vector<double> llr(M, 0);
    for (int i = 0; i < M; ++i) {
        llr[i] = log2f(joint_prob.at<double>(0, i) / joint_prob.at<double>(1, i));
    }
    vector<int> permutation = argsort<double>(llr);

    pyx_1.create(1, M, CV_64F);
    pyx_1.setTo(0);
    pyx_minus1.create(1, M, CV_64F);
    pyx_minus1.setTo(0);
    for (int i = 0; i < M; ++i) {
        pyx_1.at<double>(i) += joint_prob.at<double>(0, permutation[i]);
        pyx_minus1.at<double>(i) += joint_prob.at<double>(1, permutation[i]);
    }
    Mat table = precompute_partial_entropy_table(M, K);
    vector<int> Az(K + 1, 0);
    Az[K] = M;
    Mat state_table = Mat::zeros(M - K + 1, K + 1, CV_64F);
    Mat local_max = Mat::ones(state_table.size(), CV_32S) * (-1E20);
    for (int i = 0; i < state_table.rows; ++i) {
        state_table.at<double>(i, 1) = table.at<double>(0, 1 + i);
        local_max.at<int>(i, 1) = 0;
    }

    // dynamic programming begin
    // forward computing
    for (int z = 2; z < K + 1; ++z) {
        if (z < K){
            for (int a = z; a < z + M - K + 1; ++a) {
                int a_idx = a - z;
                int a_prime_begin = z - 1;
                int a_prime_end = a - 1;
                vector<double> tmp(a_prime_end - a_prime_begin + 1, 0);
                vector<int> tmp_idx;
                for (int a_prime = a_prime_begin, cnt = 0; a_prime < a_prime_end + 1; ++a_prime, ++cnt) {
                    tmp[cnt] = state_table.at<double>(a_prime - a_prime_begin, z - 1) + table.at<double>(a_prime, a);
                    tmp_idx.push_back(a_prime);
                }
                int max_tmp_idx = argmax(tmp.begin(), tmp.end());
                local_max.at<int>(a_idx, z) = tmp_idx[max_tmp_idx];
                state_table.at<double>(a_idx, z) = tmp[max_tmp_idx];
            }
        } else{
            int a = M;
            int a_prime_begin = z - 1;
            int a_prime_end = a - 1;
            vector<double> tmp(a_prime_end - a_prime_begin + 1, 0);
            vector<int> tmp_idx;
            for (int a_prime = a_prime_begin, cnt = 0; a_prime < a_prime_end + 1; ++a_prime, ++cnt) {
                tmp[cnt] = state_table.at<double>(a_prime - a_prime_begin, z - 1) + table.at<double>(a_prime, a);
                tmp_idx.push_back(a_prime);
            }
            int max_tmp_idx = argmax(tmp.begin(), tmp.end());
            local_max.at<int>(M - K, z) = tmp_idx[max_tmp_idx];
            state_table.at<double>(M - K, z) = tmp[max_tmp_idx];
        }
    }
    // backward tracing
    Az[K - 1] = local_max.at<int>(M - K, K);
    int opt_idx = Az[K - 1];
    for (int z = K - 1; z > 1; --z) {
        opt_idx = local_max.at<int>(opt_idx - z, z);
        Az[z - 1] = opt_idx;
    }

    // Q matrix generation
    auto Q = py::array_t<int>({K, M});
    py::buffer_info buf_result = Q.request();
    auto* ptr_Q = (int*)buf_result.ptr;
    memset(ptr_Q, 0, sizeof(int)*K*M);
    auto pzx = py::array_t<double>({2, K});
    py::buffer_info buf_pzx = pzx.request();
    auto* ptr_pzx = (double*)buf_pzx.ptr;
    memset(ptr_pzx, 0, sizeof(double)*2*K);
    for (int i = 0; i < K; ++i) {
        int begin = Az[i];
        int end = Az[i + 1];
        for (int j = begin; j < end; ++j) {
            int idx = i * M + permutation[j];
            ptr_Q[idx] = 1;
            ptr_pzx[i] += joint_prob.at<double>(0, permutation[j]);
            ptr_pzx[i+K] += joint_prob.at<double>(1, permutation[j]);
        }
    }
    return py::make_tuple(Q, pzx, Az, permutation);
}

py::tuple MMIQuantizer::find_opt_quantizer_consider_unique(py::array_t<double> &joint_prob_ndarrary, int K, const string& mode) {

    py::buffer_info buf = joint_prob_ndarrary.request();
    vector<ssize_t> size_buf = buf.shape;
    Mat joint_prob(size_buf[0], size_buf[1], CV_64F, (double*)buf.ptr);
    int M = joint_prob.cols;
    CV_Assert(M >= K);
    vector<double> llr(M, 0);
    for (int i = 0; i < M; ++i) {
        llr[i] = log2f(joint_prob.at<double>(0, i) / joint_prob.at<double>(1, i));
    }
    vector<int> permutation = argsort<double>(llr);
    vector<vector<int>> indices;
    vector<int> tmp;
    tmp.push_back(permutation[0]);
    indices.emplace_back(tmp);
    int n_unique = 1;
    for (int i = 1;  i < M; ++ i) {
        if (llr[permutation[i]] == llr[permutation[i-1]]){
            indices[n_unique - 1].push_back(permutation[i]);
        } else{
            n_unique += 1;
            vector<int> tmp1;
            tmp1.push_back(permutation[i]);
            indices.emplace_back(tmp1);
        }
    }
    pyx_1.create(1, n_unique, CV_64F);
    pyx_1.setTo(0);
    pyx_minus1.create(1, n_unique, CV_64F);
    pyx_minus1.setTo(0);
    for (int i = 0; i < n_unique; ++i) {
        for (int n = 0; n < indices[i].size(); ++n) {
            pyx_1.at<double>(i) += joint_prob.at<double>(0, indices[i][n]);
            pyx_minus1.at<double>(i) += joint_prob.at<double>(1, indices[i][n]);
        }
    }
    Mat table = precompute_partial_entropy_table(n_unique, K);
    vector<int> Az(K + 1, 0);
    Az[K] = n_unique;
    Mat state_table = Mat::zeros(n_unique - K + 1, K + 1, CV_64F);
    Mat local_max = Mat::ones(state_table.size(), CV_32S) * (-1E20);
    for (int i = 0; i < state_table.rows; ++i) {
        state_table.at<double>(i, 1) = table.at<double>(0, 1 + i);
        local_max.at<int>(i, 1) = 0;
    }

    // dynamic programming begin
    // forward computing
    for (int z = 2; z < K + 1; ++z) {
        if (z < K){
            for (int a = z; a < z + n_unique - K + 1; ++a) {
                int a_idx = a - z;
                int a_prime_begin = z - 1;
                int a_prime_end = a - 1;
                vector<double> tmp(a_prime_end - a_prime_begin + 1, 0);
                vector<int> tmp_idx;
                for (int a_prime = a_prime_begin, cnt = 0; a_prime < a_prime_end + 1; ++a_prime, ++cnt) {
                    tmp[cnt] = state_table.at<double>(a_prime - a_prime_begin, z - 1) + table.at<double>(a_prime, a);
                    tmp_idx.push_back(a_prime);
                }
                int max_tmp_idx = argmax(tmp.begin(), tmp.end());
                local_max.at<int>(a_idx, z) = tmp_idx[max_tmp_idx];
                state_table.at<double>(a_idx, z) = tmp[max_tmp_idx];
            }
        } else{
            int a = n_unique;
            int a_prime_begin = z - 1;
            int a_prime_end = a - 1;
            vector<double> tmp(a_prime_end - a_prime_begin + 1, 0);
            vector<int> tmp_idx;
            for (int a_prime = a_prime_begin, cnt = 0; a_prime < a_prime_end + 1; ++a_prime, ++cnt) {
                tmp[cnt] = state_table.at<double>(a_prime - a_prime_begin, z - 1) + table.at<double>(a_prime, a);
                tmp_idx.push_back(a_prime);
            }
            int max_tmp_idx = argmax(tmp.begin(), tmp.end());
            local_max.at<int>(n_unique - K, z) = tmp_idx[max_tmp_idx];
            state_table.at<double>(n_unique - K, z) = tmp[max_tmp_idx];
        }
    }
    // backward tracing
    Az[K - 1] = local_max.at<int>(n_unique - K, K);
    int opt_idx = Az[K - 1];
    for (int z = K - 1; z > 1; --z) {
        opt_idx = local_max.at<int>(opt_idx - z, z);
        Az[z - 1] = opt_idx;
    }

    // Q matrix generation
    auto Q = py::array_t<int>({K, M});
    py::buffer_info buf_result = Q.request();
    auto* ptr_Q = (int*)buf_result.ptr;
    memset(ptr_Q, 0, sizeof(int)*K*M);
    auto pzx = py::array_t<double>({2, K});
    py::buffer_info buf_pzx = pzx.request();
    auto* ptr_pzx = (double*)buf_pzx.ptr;
    memset(ptr_pzx, 0, sizeof(double)*2*K);
    for (int i = 0; i < K; ++i) {
        int begin = Az[i];
        int end = Az[i + 1];
        for (int j = begin; j < end; ++j) {
            for (int n = 0; n < indices[j].size(); ++n) {
                int idx = i * M + indices[j][n];
                ptr_Q[idx] = 1;
                ptr_pzx[i] += joint_prob.at<double>(0, indices[j][n]);
                ptr_pzx[i+K] += joint_prob.at<double>(1, indices[j][n]);
            }
        }
    }
    return py::make_tuple(Q, pzx, Az);
}

py::array_t<int> MMIQuantizer::find_opr_quantizer_AWGN(py::array_t<double> &joint_prob_ndarrary, int K) {
    py::buffer_info buf = joint_prob_ndarrary.request();
    vector<ssize_t> size_buf = buf.shape;
    Mat joint_prob(size_buf[0], size_buf[1], CV_64F, (double*)buf.ptr);

    int M = joint_prob.cols;
    pyx_1.create(1, M, CV_64F);
    pyx_minus1.create(1, M, CV_64F);
    memcpy(pyx_1.ptr<double>(0), joint_prob.ptr<double>(0), sizeof(double) * M);
    memcpy(pyx_minus1.ptr<double>(0), joint_prob.ptr<double>(1), sizeof(double) * M);

    Mat table = precompute_partial_entropy_table(M, K);
    vector<int> Az(K + 1, 0);
    Az[K] = M;
    Mat state_table = Mat::zeros(M - K + 1, K + 1, CV_64F);
    Mat local_max = Mat::ones(state_table.size(), CV_32S) * (-1E20);
    for (int i = 0; i < state_table.rows; ++i) {
        state_table.at<double >(i, 1) = table.at<double >(0, 1 + i);
        local_max.at<int>(i, 1) = 0;
    }
    // dynamic programming begin
    // forward computing
    for (int z = 2; z < K + 1; ++z) {
        if (z < K){
            for (int a = z; a < z + M - K + 1; ++a) {
                int a_idx = a - z;
                int a_prime_begin = z - 1;
                int a_prime_end = a - 1;
                vector<double > tmp(a_prime_end - a_prime_begin + 1, 0);
                vector<int> tmp_idx;
                for (int a_prime = a_prime_begin, cnt = 0; a_prime < a_prime_end + 1; ++a_prime, ++cnt) {
                    tmp[cnt] = state_table.at<double>(a_prime - a_prime_begin, z - 1) + table.at<double>(a_prime, a);
                    tmp_idx.push_back(a_prime);
                }
                int max_tmp_idx = argmax(tmp.begin(), tmp.end());
                local_max.at<int>(a_idx, z) = tmp_idx[max_tmp_idx];
                state_table.at<double>(a_idx, z) = tmp[max_tmp_idx];
            }
        } else{
            int a = M;
            int a_prime_begin = z - 1;
            int a_prime_end = a - 1;
            vector<double> tmp(a_prime_end - a_prime_begin + 1, 0);
            vector<int> tmp_idx;
            for (int a_prime = a_prime_begin, cnt = 0; a_prime < a_prime_end + 1; ++a_prime, ++cnt) {
                tmp[cnt] = state_table.at<double >(a_prime - a_prime_begin, z - 1) + table.at<double >(a_prime, a);
                tmp_idx.push_back(a_prime);
            }
            int max_tmp_idx = argmax(tmp.begin(), tmp.end());
            local_max.at<int>(M - K, z) = tmp_idx[max_tmp_idx];
            state_table.at<double >(M - K, z) = tmp[max_tmp_idx];
        }
    }
    // backward tracing
    Az[K - 1] = local_max.at<int>(M - K, K);
    int opt_idx = Az[K - 1];
    for (int z = K - 1; z > 1; --z) {
        opt_idx = local_max.at<int>(opt_idx - z, z);
        Az[z - 1] = opt_idx;
    }

    // copy Az to result ndarray
    auto border = py::array_t<int>(Az.size());
    py::buffer_info buf_result = border.request();
    auto* ptr_result = (int*)buf_result.ptr;
    memcpy(ptr_result, Az.data(), sizeof(int) * Az.size());
    return border;
}


