#pragma once

/*
 * Some utilities to cleanup code.
 */

#include <cstdint>

//looping
#define loop(v, m)        for(size_t v = 0; v < size_t(m); v++)
#define loopi(m)        loop(i, m)
#define loopj(m)        loop(j, m)
#define loopk(m)        loop(k, m)

/*
 * Fake destructor for C resources, to cope with exceptions. E.g.:
 * void* array = malloc(123);
 * destructor([=]{ free(array); });
 */

#define destructor(f) destructor_helper_macro_1(f, __LINE__)

#include <utility>

template<typename F>
struct destructor_helper {
	F f;

	~destructor_helper() { f(); }
};

template<typename F>
destructor_helper<F> make_destructor_helper(F&& f) {
	return destructor_helper<F>{ std::move(f) };
}

#define destructor_helper_macro_2(f, l) auto destructor_ ## l = make_destructor_helper(f)
#define destructor_helper_macro_1(f, l) destructor_helper_macro_2(f, l)

/*
 * Smallest unsigned integer that can hold a value.
 */

#include <type_traits>

template<uint64_t max, typename = void> struct smallest_uint;
template<uint64_t max> struct smallest_uint<max, std::enable_if_t<(max < 0x100)>> {
	using type = uint8_t;
};
template<uint64_t max> struct smallest_uint<max, std::enable_if_t<(max >= 0x100 && max < 0x10000)>> {
	using type = uint16_t;
};
template<uint64_t max> struct smallest_uint<max, std::enable_if_t<(max >= 0x10000 && max < 0x100000000ULL)>> {
	using type = uint32_t;
};
template<uint64_t max> struct smallest_uint<max, std::enable_if_t<(max >= 0x100000000ULL)>> {
	using type = uint64_t;
};
