#include <iostream>
#include <string>
#include <numeric>
#include <algorithm>
#include <vector>

template <typename C>
C concat(const C& lhs, const C& rhs) {
  C res(lhs.size() + rhs.size());
  typename C::iterator it = std::copy(lhs.cbegin(), lhs.cend(), res.begin());
  std::copy(rhs.cbegin(), rhs.cend(), it);
  return res;
}

template <typename T>
std::vector<T> subset(std::vector<T> const& v, unsigned int m, unsigned int n) {
  auto first = v.begin() + m;
  auto last = v.begin() + n + 1;
  std::vector<T> vector_subset(first, last);
  return vector_subset;
}

template <typename T> void print_array(T* arr, size_t len, std::string name) {
  std::cout << name << std::endl;
  for (unsigned int i = 0; i < len; i++) {
    std::cout << arr[i] << ", ";
  }
  std::cout << std::endl;
}

// Change to sorted insert...
static void arg_maxN(float* input, size_t input_len, unsigned int* output, size_t output_len) {
  unsigned int* sorted = new unsigned int[input_len];
  std::iota(sorted, sorted + input_len, 0);

  std::sort(sorted, sorted + input_len, [input](unsigned int a, unsigned int b) {
    return input[a] > input[b];
    });

  std::copy(sorted, sorted + output_len, output);
}

static unsigned int arg_max(float* input, size_t input_len) {
  unsigned int max;
  arg_maxN(input, input_len, &max, 1);
  return max;
}
