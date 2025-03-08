/// @author Sudhanva Kulkarni
/// Simplke implementation of Varying Layout dense matrix


#include <cassert>
#include "layouts.h"


namespace Lo_Gemm {

template<typename T, typename idx, Layout L = ColMajor>
class Matrix {
    private :
    idx m;
    idx n;
    idx ld;
    T* data;

    public:
    Matrix(T* data, idx m, idx n, idx ld) : data(data), m(m), n(n), ld(ld) {}

    constexpr inline T& operator() const (idx row, idx col) {
       return L == ColMajor ? data[col*ld + row] : data[row*ld + col];
    }

    bool isNaN() const {
        for(int row = 0; row < m; row++) {
            for(int col = 0; col < n; col++) {
                if(isnan((*this)(i,j))) return true;
            }
        }
        return false;
    }

    bool isInf() const {
        for(int row = 0; row < m; row++) {
            for(int col = 0; col < n; col++) {
                if(isinf((*this)(i,j))) return true;
            }
        }
        return false;
    }

    constexpr inline idx rows() const {
        return this->m;
    }

    constexpr inline idx cols() const {
        return this->n;
    }
}

template<typename T, typename idx, typename T_scal , Layout L = ColMajor, MX_Layout MX_L = byBlock>
class MX_Matrix {
    private:
    idx m;
    idx n;
    idx ld;
    idx r1; 
    idx r2;
    T* data;
    T_scal* shared_exp;

    public:
    MX_Matrix(T* data, T_scal* shared_exp, idx m, idx n, idx ld, idx r1, idx r2) : data(data), shared_exp(shared_exp), m(m), n(n), ld(ld), r1(r1), r2(r2) {
        assert((m % r1 == 0) && "Error: r1 must divide m");
        assert((n % r2 == 0) && "Error: r2 must divide n");
    }

    template<typename V>
    constexpr inline V scaled_val(idx row, idx col) const {
        if constexpr (MX_L == byBlock || MX_L == byColumn)
            return (L == ColMajor ? static_cast<V>(data[col*ld + row])*static_cast<V>(shared_exp[idx((col*ld + row)/r1)]) : 
                    static_cast<V>(data[row*ld + col])*static_cast<V>(shared_exp[idx((col + row*ld)/r1)]));
        else 
            return (L == ColMajor ? static_cast<V>(data[col*ld + row])*static_cast<V>(shared_exp[idx((col*ld + row)/r2)]) : 
            static_cast<V>(data[row*ld + col])*static_cast<V>(shared_exp[idx((col + row*ld)/r2)]));
    }


    constexpr inline T& operator() const (idx row, idx col) {
        return L == ColMajor ? data[col*ld + row] : data[row*ld + col];
    }

    bool isNaN() const {
        for(int row = 0; row < m; row++) {
            for(int col = 0; col < n; col++) {
                if(isnan((this)->scaled_val(i,j))) return true;
            }
        }
        return false;
    }

    bool isInf() const {
        for(int row = 0; row < m; row++) {
            for(int col = 0; col < n; col++) {
                if(isinf((this)->scaled_val(i,j))) return true;
            }
        }
        return false;
    }


    constexpr inline idx rows() const {
        return this->m;
    }

    constexpr inline idx cols() const {
        return this->n;
    }




}

} //namespace Lo_gemm