#include "krasnopevtseva_v_hoare_batcher_sort/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "krasnopevtseva_v_hoare_batcher_sort/common/include/common.hpp"
namespace krasnopevtseva_v_hoare_batcher_sort {

KrasnopevtsevaVHoareBatcherSortOMP::KrasnopevtsevaVHoareBatcherSortOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>();
}

bool KrasnopevtsevaVHoareBatcherSortOMP::ValidationImpl() {
  const auto &input = GetInput();
  return !input.empty();
}

bool KrasnopevtsevaVHoareBatcherSortOMP::PreProcessingImpl() {
  GetOutput() = std::vector<int>();
  return true;
}

bool KrasnopevtsevaVHoareBatcherSortOMP::RunImpl() {
  const auto &input = GetInput();
  size_t size = input.size();

  if (size <= 1) {
    GetOutput() = input;
    return true;
  }

  std::vector<int> res = input;

  int n = static_cast<int>(size);
  int numthreads = omp_get_max_threads();
  numthreads = std::min(n, numthreads);

  int thread_input_size = n / numthreads;
  int thread_input_remainder_size = n % numthreads;

  std::vector<int *> pointers(numthreads);
  std::vector<int> sizes(numthreads);
  for (int i = 0; i < numthreads; i++) {
    pointers[i] = res.data() + (i * thread_input_size);
    sizes[i] = thread_input_size;
  }
  sizes[sizes.size() - 1] += thread_input_remainder_size;

#pragma omp parallel for
  for (int i = 0; i < numthreads; i++) {
    int left = static_cast<int>(pointers[i] - res.data());
    int right = left + sizes[i] - 1;
    QuickSort(res, left, right);
  }

  BatcherMerge(thread_input_size, pointers, sizes, 32);

  GetOutput() = std::move(res);
  return true;
}

bool KrasnopevtsevaVHoareBatcherSortOMP::PostProcessingImpl() {
  return true;
}

int KrasnopevtsevaVHoareBatcherSortOMP::Partition(std::vector<int> &arr, int first, int last) {
  int i = first - 1;
  int value = arr[last];

  for (int j = first; j <= (last - 1); j++) {
    if (arr[j] <= value) {
      i++;
      std::swap(arr[i], arr[j]);
    }
  }
  std::swap(arr[i + 1], arr[last]);
  return i + 1;
}

void KrasnopevtsevaVHoareBatcherSortOMP::QuickSort(std::vector<int> &arr, int first, int last) {
  if (first < last) {
    int iter = Partition(arr, first, last);
    QuickSort(arr, first, iter - 1);
    QuickSort(arr, iter + 1, last);
  }
}

void KrasnopevtsevaVHoareBatcherSortOMP::BatcherMergeBlocksStep(int *left_pointer, int &left_size, int *right_pointer,
                                                                int &right_size) {
  std::inplace_merge(left_pointer, right_pointer, right_pointer + right_size);
  left_size += right_size;
}

void KrasnopevtsevaVHoareBatcherSortOMP::BatcherMerge(int thread_input_size, std::vector<int *> &pointers,
                                                      std::vector<int> &sizes, int par_if_greater) {
  int pack = static_cast<int>(pointers.size());
  for (int step = 1; pack > 1; step *= 2, pack /= 2) {
#pragma omp parallel for if ((thread_input_size / step) > par_if_greater)
    for (int off = 0; off < pack / 2; ++off) {
      BatcherMergeBlocksStep(pointers[2 * step * off], sizes[2 * step * off], pointers[(2 * step * off) + step],
                             sizes[(2 * step * off) + step]);
    }
    if ((pack / 2) - 1 == 0) {
      BatcherMergeBlocksStep(pointers[0], sizes[sizes.size() - 1], pointers[pointers.size() - 1],
                             sizes[sizes.size() - 1]);
    } else if ((pack / 2) % 2 != 0) {
      BatcherMergeBlocksStep(pointers[2 * step * ((pack / 2) - 2)], sizes[2 * step * ((pack / 2) - 2)],
                             pointers[2 * step * ((pack / 2) - 1)], sizes[2 * step * ((pack / 2) - 1)]);
    }
  }
}

}  // namespace krasnopevtseva_v_hoare_batcher_sort
