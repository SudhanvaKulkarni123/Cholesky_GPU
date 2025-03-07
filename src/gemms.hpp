///this file contains different variants of gemm with bfp
#include "tlapack/lapack/lacpy.hpp"
#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/plugins/legacyArray.hpp"
#include <omp.h>



int numU_to8 = 0;
int numL_to8 = 0;

template <typename Matrix>
void add_matrix(const Matrix& A, const Matrix& B, Matrix& C)
{
    using idx_t = int;
    idx_t n = A.size();
    #pragma omp parallel for collapse(2)
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            C(i, j) = A(i, j) + B(i, j);

    return;
}

// Matrix subtraction
template <typename Matrix>
void subtract_matrix(const Matrix& A, const Matrix& B, Matrix& C)
{

    using idx_t = int;
    idx_t n = A.size();

    #pragma omp parallel for collapse(2)
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            C(i, j) = A(i, j) - B(i, j);

    return;
}


template <typename matrix_t>
void PrintMatrix(const matrix_t& A)
{
    for(int i = 0; i < nrows(A); i++)
    {
        for(int j = 0; j < ncols(A); j++)
        {
            std::cout << A(i,j) << " ";
        }
        std::cout << std::endl;
    }

}

template <typename Matrix>
void strassen_multiply(Matrix& A, Matrix& B, Matrix& C,
                       Matrix& M, Matrix& tempA, Matrix& tempB, int threshold)
{
    using idx_t = tlapack::size_type<Matrix>;

    idx_t n = nrows(A); // Assuming square matrices

    if (n <= threshold)
    {
        // Standard multiplication
        #pragma omp parallel for collapse(2)
        for (idx_t i = 0; i < n; i++)
            for (idx_t j = 0; j < n; j++)
                for (idx_t k = 0; k < n; k++)
                    C(i, j) += A(i, k) * B(k, j);
        return;
    }

    idx_t newSize = n / 2;

    // Create ranges for slicing
    auto r0 = range(0, newSize);
    auto r1 = range(newSize, n);

    // Create slices for submatrices
    auto A11 = slice(A, r0, r0);
    auto A12 = slice(A, r0, r1);
    auto A21 = slice(A, r1, r0);
    auto A22 = slice(A, r1, r1);

    auto B11 = slice(B, r0, r0);
    auto B12 = slice(B, r0, r1);
    auto B21 = slice(B, r1, r0);
    auto B22 = slice(B, r1, r1);

    auto C11 = slice(C, r0, r0);
    auto C12 = slice(C, r0, r1);
    auto C21 = slice(C, r1, r0);
    auto C22 = slice(C, r1, r1);

    // Slices for M and temporary matrices
    auto M_sub = slice(M, r0, r0);
    auto tempA_sub = slice(tempA, r0, r0);
    auto tempB_sub = slice(tempB, r0, r0);

    // Use OpenMP tasks for parallelism
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            // Reuse M_sub for M1 to M7 sequentially
            // M1 = (A11 + A22) * (B11 + B22)
            add_matrix(A11, A22, tempA_sub);
            add_matrix(B11, B22, tempB_sub);
            #pragma omp task
            strassen_multiply(tempA_sub, tempB_sub, M_sub, M_sub, tempA_sub, tempB_sub, threshold);

            // M2 = (A21 + A22) * B11
            add_matrix(A21, A22, tempA_sub);
            #pragma omp task
            strassen_multiply(tempA_sub, B11, tempB_sub, M_sub, tempA_sub, tempB_sub, threshold);

            // M3 = A11 * (B12 - B22)
            subtract_matrix(B12, B22, tempB_sub);
            #pragma omp task
            strassen_multiply(A11, tempB_sub, tempA_sub, M_sub, tempA_sub, tempB_sub, threshold);

            // M4 = A22 * (B21 - B11)
            subtract_matrix(B21, B11, tempB_sub);
            #pragma omp task
            strassen_multiply(A22, tempB_sub, C11, M_sub, tempA_sub, tempB_sub, threshold);

            // M5 = (A11 + A12) * B22
            add_matrix(A11, A12, tempA_sub);
            #pragma omp task
            strassen_multiply(tempA_sub, B22, C12, M_sub, tempA_sub, tempB_sub, threshold);

            // M6 = (A21 - A11) * (B11 + B12)
            subtract_matrix(A21, A11, tempA_sub);
            add_matrix(B11, B12, tempB_sub);
            #pragma omp task
            strassen_multiply(tempA_sub, tempB_sub, C21, M_sub, tempA_sub, tempB_sub, threshold);

            // M7 = (A12 - A22) * (B21 + B22)
            subtract_matrix(A12, A22, tempA_sub);
            add_matrix(B21, B22, tempB_sub);
            #pragma omp task
            strassen_multiply(tempA_sub, tempB_sub, C22, M_sub, tempA_sub, tempB_sub, threshold);

            #pragma omp taskwait

            // Combine results directly into C submatrices
            // Note: Adjust the calculations as needed to account for the reuse of buffers
            // Ensure that the buffers used in calculations are not overwritten prematurely
        }
    }

    return;
}



/// @brief simple gemm
template<TLAPACK_MATRIX matrixA_t, TLAPACK_MATRIX matrixB_t, TLAPACK_MATRIX matrixC_t, TLAPACK_SCALAR scal_t>
void simple_gemm(matrixA_t& A, matrixB_t& B, matrixC_t& C, scal_t scale_by = 1.0)
{
    using idx_t = size_type<matrixA_t>;
    using T = type_t<matrixA_t>;
    const idx_t m = nrows(A);
    const idx_t n = ncols(B);
    const idx_t k = ncols(A);
    for (idx_t i = 0; i < m; i++)
    {
        for (idx_t j = 0; j < n; j++)
        {
            float sum = 0;
            for (idx_t l = 0; l < k; l++)
            {
                sum += scale_by * float(A(i, l)) * float(B(l, j));
            }
            C(i, j) -= sum;
        }
    }
    return;
}






/// @brief preforms C = C + AB using mocroscaling format. A gets common exp over rows and B gets common exp over columns. C is in whatever format the original matrix is in
/// @tparam matrix_t 
/// @param A 
/// @param B 
/// @param C 
template<TLAPACK_MATRIX matrixA_t, TLAPACK_MATRIX matrixB_t, TLAPACK_MATRIX matrixC_t>
void block_gemm(matrixA_t& A, matrixA_t& B, matrixA_t& C, matrixB_t& A_dtype, matrixC_t& B_dtype, int block_size = 4)
{
    using idx_t = size_type<matrixA_t>;
    using T = type_t<matrixA_t>;
    using real_t = real_type<T>;
    using V = type_t<matrixB_t>;
    using dtype = real_type<V>;
    using U = type_t<matrixC_t>;
    using dtype2 = real_type<U>;
    const idx_t m = nrows(A);
    const idx_t n = ncols(B);
    const idx_t k = ncols(A);



    
    //first find max exp in all rows for A
    std::vector<int> max_exp_A1(m, 0);
    std::vector<int> max_exp_A2(m, 0);
    // for (idx_t i = 0; i < m; i++)
    // {   
    //     if(blocks){
    //     int max_exp = -999;
    //     for (idx_t j = 0; j < k; j++)
    //     {
    //         if (int(floor(log2(abs(A(i, j))))) > max_exp) max_exp = int(floor(log2(abs(A(i, j)))));
    //     }
    //     max_exp_A1[i] = 0;
    //     } else {
    //         max_exp_A1[i] = 0;
    //     }
    // }

    //now find max exp in all columns for B
    std::vector<int> max_exp_B1(n, 0);
    // for (idx_t j = 0; j < n; j++)
    // {   
    //     if(blocks){
    //     int max_exp = -999;
    //     for (idx_t i = 0; i < k; i++)
    //     {
    //         if (int(floor(log2(abs(B(i, j))))) > max_exp) max_exp = int(floor(log2(abs(B(i, j)))));
    //     }
    //     max_exp_B1[j] = 0;
    //     } else {
    //         max_exp_B1[j] = 0;
    //     }
    // }

    //now scale A and B accordingly -- A is lower triangular, B upper
    #pragma omp parallel for 
    for (idx_t j = 0; j < k; j++)
    {
        #pragma omp simd
        for (idx_t i = 0; i < m; i++)
        {
            A_dtype(i, j) = static_cast<dtype>((A(i, j)/std::pow(2.0,max_exp_A1[i])));
        }
    }

    #pragma omp parallel for 
    for (idx_t j = 0; j < n; j++)
    {
        #pragma omp simd
        for (idx_t i = 0; i < k; i++)
        {
            B_dtype(i, j) = static_cast<dtype2>(B(i, j)/ std::pow(2.0,max_exp_B1[j]));
        }
    }

    //now perform the matmul as would be done in tensor cores
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (idx_t ii = 0; ii < m; ii += block_size)
{
    for (idx_t jj = 0; jj < n; jj += block_size)
    {
        for (idx_t ll = 0; ll < k; ll += block_size)
        {
            for (idx_t i = ii; i < std::min(ii + block_size, m); i++)
            {
                for (idx_t j = jj; j < std::min(jj + block_size, n); j++)
                {
                    float sum = 0;
                    #pragma omp simd reduction(+:sum)
                    for (idx_t l = ll; l < std::min(ll + block_size, k); l++)
                    {
                    //     auto first_leftover = static_cast<dtype>(std::pow(2, 7) * 
                    //         (static_cast<float>(A(i, l)) - std::pow(2.0, max_exp_A1[i]) * static_cast<float>(A_dtype(i, l))));
                    //     auto second_leftover = static_cast<dtype2>(std::pow(2, 7) * 
                    //         (static_cast<float>(B(l, j)) - std::pow(2.0, max_exp_B1[j]) * static_cast<float>(B_dtype(l, j))));

                        sum += static_cast<float>(A_dtype(i, l)) * static_cast<float>(B_dtype(l, j));
                        
                        // sum += (static_cast<float>(std::pow(2, 7)) * static_cast<float>(A_dtype(i, l)) * static_cast<float>(B_dtype(l, j)) +
                        //         (static_cast<float>(first_leftover) * static_cast<float>(B_dtype(l, j)) +
                        //          static_cast<float>(A_dtype(i, l)) * static_cast<float>(second_leftover))) / std::pow(2, 7);
                    }
                    C(i, j) -= sum;
                }
            }
        }
    }
}

    
    // Free the vectors max_exp_A and max_exp_B



    return;

}



/// @brief preforms C = C + AB using mocroscaling format. A and B get common exp over a 4-by-4 block 
/// @tparam matrix_t
/// @param A
/// @param B
/// @param C
template<TLAPACK_MATRIX matrixA_t, TLAPACK_MATRIX matrixB_t>
void fbfmatmul(matrixA_t& A, matrixA_t& B, matrixA_t& C, matrixB_t& A_dtype, matrixB_t& B_dtype, int block_size = 2)
{

    using idx_t = size_type<matrixA_t>;
    using T = type_t<matrixA_t>;
    using real_t = real_type<T>;
    using V = type_t<matrixB_t>;
    using dtype = real_type<V>;
    using range = std::pair<idx_t, idx_t>;
    const idx_t m = nrows(A);
    const idx_t n = ncols(B);
    const idx_t k = ncols(A);


    for(int i = 0; i < m/block_size; i++) 
    {
        for(int j = 0; j < n/block_size; j++) 
        {
            auto C_p = tlapack::slice(C, range(i*block_size, (i+1)*block_size), range(j*block_size, (j+1)*block_size));
            for(int l = 0; l < k/block_size; l++){

                auto A_p = tlapack::slice(A, range(i*block_size, (i+1)*block_size), range(l*block_size, (l+1)*block_size));
                auto B_p = tlapack::slice(B, range(l*block_size, (l+1)*block_size), range(j*block_size, (j+1)*block_size));
                
                auto Adp = tlapack::slice(A_dtype, range(0, (1)*block_size), range(0, (1)*block_size));
                auto Bdp = tlapack::slice(B_dtype, range(0, block_size), range(0, block_size));
                //take exp common
                int max_exp_A = -999;
                int max_exp_B = -999;
                for(int ii = 0; ii < block_size; ii++){
                    for(int jj = 0; jj < block_size; jj++) {
                        if (int(floor(log2(abs(A_p(ii, jj))))) > max_exp_A) max_exp_A = int((log2(abs(A_p(ii, jj)))));
                        if (int(floor(log2(abs(B_p(ii, jj))))) > max_exp_B) max_exp_B = int((log2(abs(B_p(ii, jj)))));
                    }
                }
                max_exp_A -= 2;
                max_exp_B -= 2;


                for(int ii = 0; ii < block_size; ii++){
                    for(int jj = 0; jj < block_size; jj++) {
                        Adp(ii, jj) = static_cast<dtype>(A_p(ii, jj)/std::pow(2.0,max_exp_A));
                        Bdp(ii, jj) = static_cast<dtype>(B_p(ii, jj)/std::pow(2.0,max_exp_B));
                    }
                }
                simple_gemm(Adp, Bdp, C_p, std::pow(2.0,max_exp_A + max_exp_B));

            }
            
        }
    }


    return;


}


/// @brief preforms C = C + AB using mocroscaling format. updates on diagonal are done in fp16, off-diagonal in fp8
template<TLAPACK_MATRIX matrixA_t, TLAPACK_MATRIX matrixB_t, TLAPACK_MATRIX matrixC_t>
void fbfmatmul_fp16(matrixA_t& A, matrixA_t& B, matrixA_t& C, matrixB_t& A_dtype, matrixB_t& B_dtype, matrixC_t& o_A_dtype, matrixC_t& o_B_dtype, int block_size = 4)
{

    using idx_t = size_type<matrixA_t>;
    using T = type_t<matrixA_t>;
    using real_t = real_type<T>;
    using V = type_t<matrixB_t>;
    using dtype = real_type<V>;
    using range = std::pair<idx_t, idx_t>;
    using odtype = real_type<type_t<matrixC_t>>;
    const idx_t m = nrows(A);
    const idx_t n = ncols(B);
    const idx_t k = ncols(A);




    
    //first find max exp in all rows for A
    std::vector<int> max_exp_A1(m, 0);
    std::vector<int> max_exp_A2(m, 0);
    for (idx_t i = 0; i < m; i++)
    {   
        if(blocks){
        int max_exp = -999;
        for (idx_t j = 0; j < k; j++)
        {
            if (int(floor(log2(abs(A(i, j))))) > max_exp) max_exp = int(floor(log2(abs(A(i, j)))));
        }
        max_exp_A1[i] = 0;
        } else {
            max_exp_A1[i] = 0;
        }
    }

    //now find max exp in all columns for B
    std::vector<int> max_exp_B1(n, 0);
    for (idx_t j = 0; j < n; j++)
    {   
        if(blocks){
        int max_exp = -999;
        for (idx_t i = 0; i < k; i++)
        {
            if (int(floor(log2(abs(B(i, j))))) > max_exp) max_exp = int(floor(log2(abs(B(i, j)))));
        }
        max_exp_B1[j] = 0;
        } else {
            max_exp_B1[j] = 0;
        }
    }

    //now scale A and B accordingly -- A is lower triangular, B upper
    for (idx_t i = 0; i < m; i++)
    {
        for (idx_t j = 0; j < k; j++)
        {
            A_dtype(i, j) = static_cast<dtype>((A(i, j)/std::pow(2.0,max_exp_A1[i])));
        }
    }
    for (idx_t i = 0; i < m; i++)
    {
        for (idx_t j = 0; j < k; j++)
        {
            o_A_dtype(i, j) = static_cast<odtype>((A(i, j)/std::pow(2.0,max_exp_A1[i])));
        }
    }

    for (idx_t i = 0; i < k; i++)
    {
        for (idx_t j = 0; j < n; j++)
        {
            B_dtype(i, j) = static_cast<dtype>(B(i, j)/ std::pow(2.0,max_exp_B1[j]));
        }
    }
    for (idx_t i = 0; i < k; i++)
    {
        for (idx_t j = 0; j < n; j++)
        {
            o_B_dtype(i, j) = static_cast<odtype>(B(i, j)/ std::pow(2.0,max_exp_B1[j]));
        }
    }



    //now perform the matmul as would be done in tensor cores
    for (idx_t i = 0; i < m; i++)
    {
        for (idx_t j = 0; j < n; j++)
        {
            float sum = 0;
            for (idx_t l = 0; l < k; l++)
            {
                auto first_leftover  = static_cast<dtype>(std::pow(2,7)*(static_cast<float>(A(i,l)) - std::pow(2.0,max_exp_A1[i])*static_cast<float>(A_dtype(i,l))));
                auto second_leftover = static_cast<dtype>(std::pow(2,7)*(static_cast<float>(B(l,j)) - std::pow(2.0,max_exp_B1[j])*static_cast<float>(B_dtype(l,j))));
                
                if(i >= j-4 && i <= j+4){ 
                    sum += static_cast<float>(o_A_dtype(i, l)) * static_cast<float>(o_B_dtype(l, j));
                } else {
                    //sum += static_cast<float>(static_cast<float>(std::pow(2,7))*static_cast<float>(A_dtype(i, l)) * static_cast<float>(B_dtype(l, j)) + (static_cast<float>((first_leftover)) * static_cast<float>(B_dtype(l, j)) + static_cast<float>(A_dtype(i, l)) * static_cast<float>((second_leftover))))/std::pow(2, 7);
                    // + std::pow(2,-14)*(static_cast<float>(first_leftover) * static_cast<float>(second_leftover));
                    sum += static_cast<float>(A_dtype(i, l)) * static_cast<float>(B_dtype(l, j));
                }
            }
            C(i, j) -= sum*std::pow(2.0,max_exp_A1[i] + max_exp_B1[j]);
        }
    }
   
    // Free the vectors max_exp_A and max_exp_B





    return;
}




template<TLAPACK_MATRIX matrixA_t, TLAPACK_MATRIX matrixB_t, TLAPACK_MATRIX matrixC_t>
void squeezing_matmul(matrixA_t& A, matrixA_t& B, matrixC_t& C, matrixB_t& A_dtype, matrixB_t& B_dtype, float z1, float z2, int block_size = 4)
{

    using idx_t = size_type<matrixA_t>;
    using T = type_t<matrixA_t>;
    using real_t = real_type<T>;
    using V = type_t<matrixB_t>;
    using dtype = real_type<V>;
    using range = std::pair<idx_t, idx_t>;
    const idx_t m = nrows(A);
    const idx_t n = ncols(B);
    const idx_t k = ncols(A);

    //find signed max and min of A, B
    float max_A = -999;
    float min_A = 999;
    float max_B = -999;
    float min_B = 999;

    #pragma omp parallel for reduction(max : max_A) collapse(2)
    for (idx_t j = 0; j < k; j++)
    {
        for (idx_t i = 0; i < m; i++)
        {
            if ((A(i, j)) > max_A) max_A = (A(i, j));
        }
    }

    //find signed max and min in B
    #pragma omp parallel for reduction(max : max_B) collapse(2)
    for (idx_t j = 0; j < n; j++)
    {

        for (idx_t i = 0; i < k; i++)
        {
            if ((B(i, j)) > max_B) max_B = (B(i, j));
        }
    }

    auto alpha1 = 1.0/max_A;
    auto alpha2 =  1.0/max_B;

    auto beta1 = 0.0;
    auto beta2 = 0.0;

    // squueze into A_dtype and B_dtype
    #pragma omp parallel for
    for (idx_t j = 0; j < k; j++)
    {
        #pragma omp simd
        for (idx_t i = 0; i < m; i++)
        {
            auto tmp = alpha1*A(i, j) + beta1;
            A_dtype(i, j) = static_cast<dtype>(tmp);
            
        }
    }

    std::vector<float> B_sums(n, 0);

    #pragma omp parallel for 
    for (idx_t j = 0; j < n; j++)
    {
        #pragma omp simd
        for (idx_t i = 0; i < k; i++)
        {
            auto tmp = alpha2*B(i, j) + beta2;
            B_dtype(i, j) = static_cast<dtype>(tmp);
        }
    }

    std:cout << "alpha 1 is ; " << alpha1 << "\n";
    std::cout << "alpha2 is : " << alpha2 << "\n";


    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int jj = 0; jj < n; jj += block_size) {
    for (int ii = 0; ii < m; ii += block_size) {
        for (int kk = 0; kk < k; kk += block_size) {
            for (int j = jj; j < jj + block_size; j++) {
                for (int i = ii; i < ii + block_size; i++)  {
                    float sum = 0;
                    #pragma omp simd reduction(+:sum)
                    for (int l = kk; l < kk + block_size; l++) {
                        sum += (static_cast<float>(A_dtype(i, l)) * static_cast<float>(B_dtype(l, j)));
                    }
                    if (kk + block_size >= k) { // Only scale and subtract in the final kk loop
                        sum = sum / (alpha1 * alpha2);
                        C(i, j) -= sum;
                    }
                }
            }
            }
        }
        }

 


    return;

    

    



}


template<TLAPACK_MATRIX matrixA_t, TLAPACK_MATRIX matrixB_t, TLAPACK_MATRIX matrixC_t>
void MicroSqueeze_matmul(matrixA_t& A, matrixA_t& B, matrixC_t& C, matrixB_t& A_dtype, matrixB_t& B_dtype, float z1, float z2, int block_size = 4)
{

    using idx_t = size_type<matrixA_t>;
    using T = type_t<matrixA_t>;
    using real_t = real_type<T>;
    using V = type_t<matrixB_t>;
    using dtype = real_type<V>;
    using range = std::pair<idx_t, idx_t>;
    const idx_t m = nrows(A);
    const idx_t n = ncols(B);
    const idx_t k = ncols(A);

    //block pattern is assumed to be the same as tiling pattern in this functyion.  
    //TODO - support different blocking pattterns through enum/ OCP class

    int mm = (int)m/block_size;
    int nn = (int)n/block_size;
    int kk = (int)k/block_size;



    

    //find max for each block

    for(int i = 0; i < mm; i++) {
        for(int j = 0; j < nn; j++) {

            for(int ii = 0; ii < block_size; ii++) {
                for(int jj = 0; jj < block_size ; j++) {

                    
                }
            }

        }
    }








    
}


template<TLAPACK_MATRIX matrixA_t, TLAPACK_MATRIX matrixB_t, TLAPACK_MATRIX matrixC_t>
void squeezing_strassen(matrixA_t& A, matrixA_t& B, matrixC_t& C, matrixB_t& A_dtype, matrixB_t& B_dtype, matrixC_t& M, matrixC_t& tempA, matrixC_t& tempB, float z1, float z2, int block_size = 4)
{
    using idx_t = size_type<matrixA_t>;
    using T = type_t<matrixA_t>;
    using real_t = real_type<T>;
    using V = type_t<matrixB_t>;
    using dtype = real_type<V>;
    using range = std::pair<idx_t, idx_t>;
    const idx_t m = nrows(A);
    const idx_t n = ncols(B);
    const idx_t k = ncols(A);

    //find signed max and min of A, B
    float max_A = -999;
    float min_A = 999;
    float max_B = -999;
    float min_B = 999;


    #pragma omp parallel for reduction(max : max_A) collapse(2)
    for (idx_t j = 0; j < k; j++)
    {
        for (idx_t i = 0; i < m; i++)
        {
            if ((A(i, j)) > max_A) max_A = (A(i, j));
        }
    }

    //find signed max and min in B
    #pragma omp parallel for reduction(max : max_B) collapse(2)
    for (idx_t j = 0; j < n; j++)
    {
        for (idx_t i = 0; i < k; i++)
        {
            if ((B(i, j)) > max_B) max_B = (B(i, j));
        }
    }

    auto alpha1 = 1.0/max_A;
    auto alpha2 =  1.0/max_B;

    auto beta1 = 0.0;
    auto beta2 = 0.0;

    // squueze into A_dtype and B_dtype
    #pragma omp parallel for
    for (idx_t j = 0; j < k; j++)
    {
        #pragma omp simd
        for (idx_t i = 0; i < m; i++)
        {
            auto tmp = alpha1*A(i, j) + beta1;
            A_dtype(i, j) = static_cast<dtype>(tmp);
            
        }
    }

    std::vector<float> B_sums(n, 0);
    #pragma omp parallel for
    for (idx_t j = 0; j < n; j++)
    {
        #pragma omp simd
        for (idx_t i = 0; i < k; i++)
        {
            auto tmp = alpha2*B(i, j) + beta2;
            B_dtype(i, j) = static_cast<dtype>(tmp);
        }
    }

    strassen_multiply(A_dtype, B_dtype, C, M, tempA, tempB, (int)(block_size/4));

    // Adjust the result and subtract from C
    #pragma omp parallel for collapse(2)
    for (idx_t i = 0; i < m; i++)
    {
        for (idx_t j = 0; j < n; j++)
        {
            float sum = C_pad(i, j) / (alpha1 * alpha2);
            C(i, j) -= sum;
        }
    }







}

template<int p, typename matrixA_t, typename matrixB_t, typename matrixC_t>
void diff_matmul(matrixA_t& A, matrixA_t& B, matrixC_t& C, matrixB_t& A_dtype, matrixB_t& B_dtype, float z1, float z2, int block_size = 4)
{
    using idx_t = size_type<matrixA_t>;
    using T = type_t<matrixA_t>;
    using real_t = real_type<T>;
    using V = type_t<matrixB_t>;
    using dtype = real_type<V>;
    using range = std::pair<idx_t, idx_t>;
    const idx_t m = nrows(A);
    const idx_t n = ncols(B);
    const idx_t k = ncols(A);

    //find signed max and min of A, B
    float max_A = -999;
    float min_A = 999;
    float max_B = -999;
    float min_B = 999;


    #pragma omp parallel for reduction(max : max_A) collapse(2)
    for (idx_t j = 0; j < k; j++)
    {
        for (idx_t i = 0; i < m; i++)
        {
            if ((A(i, j)) > max_A) max_A = (A(i, j));
        }
    }

    //find signed max and min in B
    #pragma omp parallel for reduction(max : max_B) collapse(2)
    for (idx_t j = 0; j < n; j++)
    {
        for (idx_t i = 0; i < k; i++)
        {
            if ((B(i, j)) > max_B) max_B = (B(i, j));
        }
    }

    auto alpha1 = 1.0/max_A;
    auto alpha2 =  1.0/max_B;

    auto beta1 = 0.0;
    auto beta2 = 0.0;

    // squueze into A_dtype and B_dtype
    #pragma omp parallel for
    for (idx_t j = 0; j < k; j++)
    {
        #pragma omp simd
        for (idx_t i = 0; i < m; i++)
        {
            auto tmp = alpha1*A(i, j) + beta1;
            A_dtype(i, j) = static_cast<dtype>(tmp);
            
        }
    }

    std::vector<float> B_sums(n, 0);
    #pragma omp parallel for
    for (idx_t j = 0; j < n; j++)
    {
        #pragma omp simd
        for (idx_t i = 0; i < k; i++)
        {
            auto tmp = alpha2*B(i, j) + beta2;
            B_dtype(i, j) = static_cast<dtype>(tmp);
        }
    }
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int jj = 0; jj < n; jj += block_size) {
    for (int ii = 0; ii < m; ii += block_size) {
        for (int kk = 0; kk < k; kk += block_size) {
            for (int j = jj; j < jj + block_size; j++) {
                for (int i = ii; i < ii + block_size; i++)  {
                    float sum = 0;
                    #pragma omp simd reduction(+:sum)
                    for (int l = kk; l < kk + block_size; l++) {
                        auto first_leftover = static_cast<dtype>(std::pow(2, p+1)*(A(i, l) - static_cast<float>(A_dtype(i, l))/alpha1));
                        auto second_leftover = static_cast<dtype>(std::pow(2, p+1)*(B(l, j) - static_cast<float>(B_dtype(l, j))/alpha2));
                        sum += (static_cast<float>(A_dtype(i, l)) * static_cast<float>(B_dtype(l, j)))/(alpha1*alpha2) + std::pow(2.0, -p-1) * (static_cast<float>(first_leftover)*static_cast<float>(B_dtype(l, j))/alpha2 + static_cast<float>(A_dtype(i, l))*static_cast<float>(second_leftover)/alpha1);
                    }
                    if (kk + block_size >= k) { // Only scale and subtract in the final kk loop
                        C(i, j) -= sum;
                    }
                }
            }
        }
    }
}





}


template<typename matrixA_t, typename matrixB_t, typename matrixC_t, typename mask_t>
void sparse_squeezing_matmul(matrixA_t& A, matrixA_t& B, matrixC_t& C, matrixB_t& A_dtype, matrixB_t& B_dtype, mask_t masks, float z1, float z2,float dropping_prob=0.5,  int block_size = 4)
{
    using idx_t = size_t;
   using T = type_t<matrixA_t>;
    using real_t = real_type<T>;
    using V = type_t<matrixB_t>;
    using dtype = real_type<V>;
    using range = std::pair<idx_t, idx_t>;
    const idx_t m = nrows(A);
    const idx_t n = ncols(B);
    const idx_t k = ncols(A);

    // Find signed max and min of A
    float max_A = -std::numeric_limits<float>::infinity();
    float min_A = std::numeric_limits<float>::infinity();
    for (idx_t i = 0; i < m; i++)
    {
        for (idx_t j = 0; j < k; j++)
        {
            if (abs(A(i,j)) > max_A) max_A = abs(A(i,j));
            if (A(i,j) < min_A) min_A = A(i,j);
        }
    }

    // Find signed max and min in B
    float max_B = -std::numeric_limits<float>::infinity();
    float min_B = std::numeric_limits<float>::infinity();
    for (idx_t i = 0; i < k; i++)
    {
        for (idx_t j = 0; j < n; j++)
        {
            if (abs(B(i,j)) > max_B) max_B = abs(B(i,j));
            if (B(i,j) < min_B) min_B = B(i,j);
        }
    }

    auto alpha1 = 1.0/max_A;
    auto alpha2 =  1.0/max_B;

    std::cout << "alpha1 is : " << alpha1 << "\n";
    std::cout << "alpha2 is : " << alpha2 << "\n";

    auto beta1 = 0.0;
    auto beta2 = 0.0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution bernoulli_dist(dropping_prob); // Probability to keep a term

    if(masks.size() < 64) {
        for(int i = 0; i < masks.size(); i++) std::cout << masks[i] << ", ";
        std::cout << "\n";
    }

    // Squeeze into A_dtype and B_dtype
    std::vector<float> A_sums(m, 0);
    for (idx_t i = 0; i < m; i++)
    {
        for (idx_t j = 0; j < k; j++)
        {
            auto tmp = alpha1 * A(i,j) + beta1;
            if (tmp > (float)std::numeric_limits<dtype>::max()) tmp = (float)std::numeric_limits<dtype>::max();
            if(bernoulli_dist(gen)) A_dtype(i,j) = static_cast<dtype>(0.0);
            else A_dtype(i,j) = static_cast<dtype>(tmp);
            if (isinf(A_dtype(i,j))) std::cout << "Encountered infinity in A!" << std::endl;
            A_sums[i] += alpha1 * A(i,j) + beta1;
        }
    }

    std::vector<float> B_sums(n, 0);
    for (idx_t i = 0; i < k; i++)
    {
        for (idx_t j = 0; j < n; j++)
        {
            B_dtype(i,j) = A_dtype(j,i);
        }
    }

    // Initialize random number generator for dropping terms


    for (idx_t ii = 0; ii < m; ii += block_size)
{
    for (idx_t jj = 0; jj < n; jj += block_size)
    {
        for (idx_t ll = 0; ll < k; ll += block_size)
        {

            for (idx_t i = ii; i < std::min(ii + block_size, m); i++)
            {
                for (idx_t j = jj; j < std::min(jj + block_size, n); j++)
                {
                    float sum = 0;
                    for (idx_t l = ll; l < std::min(ll + block_size, k); l++)
                    {
                        sum += static_cast<float>(A_dtype(i, l)) * static_cast<float>(B_dtype(l, j));
                    }

                    C(i, j) -= sum/(alpha1*alpha2);
                }
            }
        }
    }
}


    return;
}


template<TLAPACK_MATRIX matrixA_t, TLAPACK_MATRIX matrixB_t, TLAPACK_MATRIX matrixC_t>
void scaled_matmul(matrixA_t& A, matrixA_t& B, matrixC_t& C, matrixB_t& A_dtype, matrixB_t& B_dtype, int block_size=4)
{

    using idx_t = size_type<matrixA_t>;
    using T = type_t<matrixA_t>;
    using real_t = real_type<T>;
    using V = type_t<matrixB_t>;
    using dtype = real_type<V>;
    using range = std::pair<idx_t, idx_t>;
    const idx_t m = nrows(A);
    const idx_t n = ncols(B);
    const idx_t k = ncols(A);

    int num_A_blocks = m/block_size;
    int num_B_blocks = n/block_size;
    std::vector<float> min_A(num_A_blocks, INFINITY);
    std::vector<float> min_B(num_B_blocks, INFINITY);


    for(idx_t t = 0; t < num_A_blocks; t++) 
    {
        for (idx_t i = 0; i < block_size; i++)
        {
            for (idx_t j = 0; j < block_size; j++)
            {
                if (abs(A(t*block_size + i, j)) < min_A[t]) min_A[t] = abs(A(t*block_size + i, j));
            }
        }

    }

    for(idx_t t = 0; t < num_B_blocks; t++) 
    {
        for (idx_t i = 0; i < block_size; i++)
        {
            for (idx_t j = 0; j < block_size; j++)
            {
                if (abs(B(i, t*block_size + j)) < min_B[t]) min_B[t] = abs(B(i, t*block_size + j));
            }
        }

    }

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < k; j++) {
            A_dtype(i,j) = static_cast<dtype>(A(i,j)/min_A[i/block_size]);
        }
    }

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < k; j++) {
            B_dtype(j, i) = static_cast<dtype>(B(j,i)/min_B[i/block_size]);
        }

    }


        


    for(int i = 0; i < m; i++) 
    {
        for(int j = 0; j < n; j++) 
        {
            float sum = 0;
            for(int l = 0; l < k; l++)
            {
                sum += (min_A[i/block_size]*min_B[j/block_size]*static_cast<float>(A_dtype(i, l)) * static_cast<float>(B_dtype(l, j)));
            }
            std::cout << "sum is : " << sum << "\n";
            C(i, j) -= sum;
        }
    }


    return;




}




template<TLAPACK_MATRIX matrixA_t, TLAPACK_MATRIX matrixB_t, TLAPACK_MATRIX matrixC_t>
void multi_block_gemm(matrixA_t& A, matrixA_t& B, matrixA_t& C, matrixB_t& A_dtype, matrixB_t& B_dtype, matrixC_t& A_dtype2, matrixC_t& B_dtype2, float normA = 1.0, double eps = 1.0/256.0, int block_size = 4)
{
    
        using idx_t = size_type<matrixA_t>;
        using T = type_t<matrixA_t>;
        using real_t = real_type<T>;
        using V = type_t<matrixB_t>;
        using dtype = real_type<V>;
        using W = type_t<matrixC_t>;
        using dtype2 = real_type<W>;
        using range = std::pair<idx_t, idx_t>;
        const idx_t m = nrows(A);
        const idx_t n = ncols(B);
        const idx_t k = ncols(A);
        bool A_to_8 = false;
        bool B_to_8 = false;

        for(int i = 0; i < m/4; i++) 
        {
            for(int j = 0; j < n/4 ; j++)
            {
                auto C_p = tlapack::slice(C, range(i*4, (i+1)*4), range(j*4, (j+1)*4));
                for(int l = 0; l < k/4; l++)
                {
                    auto A_p = tlapack::slice(A, range(i*4, (i+1)*4), range(l*4, (l+1)*4));
                    auto B_p = tlapack::slice(B, range(l*4, (l+1)*4), range(j*4, (j+1)*4));
                    A_to_8 = tlapack::lange(tlapack::Norm::Fro, A_p) < 32.0*eps*normA;
                    B_to_8 = tlapack::lange(tlapack::Norm::Fro, B_p) < 32.0*eps*normA;
                    for(int ii = 0; ii < 4; ii++)
                    {
                        for(int jj = 0; jj < 4; jj++)
                        {   
                            if(A_to_8) A_dtype(ii, jj) = static_cast<dtype>(A_p(ii, jj));
                            else A_dtype2(ii, jj) = static_cast<dtype2>(A_p(ii, jj));

                            if(B_to_8) B_dtype(ii, jj) = static_cast<dtype>(B_p(ii, jj));
                            else B_dtype2(ii, jj) = static_cast<dtype2>(B_p(ii, jj));
                        }
                    }

                    if(A_to_8 && B_to_8) {
                        numL_to8++;
                        numU_to8++;
                        simple_gemm(A_dtype, B_dtype, C_p, 1.0);
                    } else if(A_to_8 && !B_to_8) {
                        numL_to8++;
                        simple_gemm(A_dtype, B_dtype2, C_p, 1.0);
                    } else if(!A_to_8 && B_to_8) {
                        numU_to8++;
                        simple_gemm(A_dtype2, B_dtype, C_p, 1.0);
                    } else {
                        simple_gemm(A_dtype2, B_dtype2, C_p, 1.0);
                    }
                    

                }
            }
        }
        return;
}