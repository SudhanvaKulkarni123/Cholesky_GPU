///@author Sudhanva Kulkarni
///File for single and half precision cholesky microkernels (n <= 64)


#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>


// function for cholesky on n <= 32 Fp32 matrices, assumes matrix in col major
//no prefetching for now, will try again if more performance is needed -- premature optimization is the root of all evil!
int micro_cholesky(float* A, int n, int ld) {
    //quick return
    if(n == 0 ) return 0;

     auto A_ = [&](int i, int j) -> float& {
        return A[j * ld + i];
    };
    
    
    for (int k = 0; k < n; ++k) {
        if(A_(k,k) < 0) std::cout << "diag entry less than 0, matrix not spd\n";

        A_(k,k) = std::sqrt(A_(k,k));
        auto A_kk = A_(k,k);

        if( A_kk == 0.0) std::cout << "encountred 0 in cholesky!\n"; 

        for (int i = k + 1; i < n; ++i) {
            A_(i,k) /= A_kk;
        }

        for (int j = k + 1; j < n; ++j) {
            for (int i = j; i < n; ++i) {
                A_(i,j) -= A_(i,k) * A_(j,k);
            }
        }
    }


    return 0;
}
