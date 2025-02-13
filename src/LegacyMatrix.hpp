



#include <cassert>

enum Layout : uint8_t {
    RowMajor,
    ColMajor
}




/** Legacy matrix.
 *
 * LegacyMatrix::ldim is assumed to be positive.
 *
 * @tparam T Floating-point type
 * @tparam idx_t Index type
 * @tparam L Either Layout::ColMajor or Layout::RowMajor
 */
template <class T,
          class idx_t = std::size_t,
          Layout L = Layout::ColMajor,
          std::enable_if_t<(L == Layout::RowMajor) || (L == Layout::ColMajor),
                           int> = 0>
struct LegacyMatrix {
    idx_t m, n;  ///< Sizes
    T* ptr;      ///< Pointer to array in memory
    idx_t ldim;  ///< Leading dimension

    static constexpr Layout layout = L;

    constexpr const T& operator()(idx_t i, idx_t j) const noexcept
    {
        assert(i >= 0);
        assert(i < m);
        assert(j >= 0);
        assert(j < n);
        return (layout == Layout::ColMajor) ? ptr[i + j * ldim]
                                            : ptr[i * ldim + j];
    }

    constexpr T& operator()(idx_t i, idx_t j) noexcept
    {
        assert(i >= 0);
        assert(i < m);
        assert(j >= 0);
        assert(j < n);
        return (layout == Layout::ColMajor) ? ptr[i + j * ldim]
                                            : ptr[i * ldim + j];
    }

    constexpr LegacyMatrix(idx_t m, idx_t n, T* ptr, idx_t ldim)
        : m(m), n(n), ptr(ptr), ldim(ldim)
    {
        tlapack_check(m >= 0);
        tlapack_check(n >= 0);
        tlapack_check(ldim >= ((layout == Layout::ColMajor) ? m : n));
    }

    constexpr LegacyMatrix(idx_t m, idx_t n, T* ptr)
        : m(m), n(n), ptr(ptr), ldim((layout == Layout::ColMajor) ? m : n)
    {
        tlapack_check(m >= 0);
        tlapack_check(n >= 0);
    }
};


