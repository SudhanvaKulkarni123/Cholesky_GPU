///@author Sudhanva Kulkarni
///File for single and half precision cholesky microkernels (n <= 64)


#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

using FP32_matrix_t = LegacyMatrix<float, int, is_symm>;

// function for cholesky on n <= 32 Fp32 matrices, assumes matrix in col major
int micro_cholesky(vector<float>& A, int n, int ld) {
    //quick return
    if(n == 0 ) return 0;

     auto A_ = [&](int i, int j) -> float& {
        return A[i * ld + j];
    };
    
    
    for (int k = 0; k < n; ++k) {

        A_(k,k) = std::sqrt(A_(k,k));
        auto A_kk = A_(k,k);

        for (int i = k + 1; i < n; ++i) {
            A_(i,k) /= Akk;
        }

        for (int j = k + 1; j < n; ++j) {
            for (int i = j; i < n; ++i) {
                A_(i,j) -= A_(i,k) * A_(j,k);
            }
        }
    }


    return 0;
}
