#include "pikhotskiy_r_vertical_gauss_filter/seq/include/ops_seq.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "pikhotskiy_r_vertical_gauss_filter/common/include/common.hpp"

namespace pikhotskiy_r_vertical_gauss_filter {

namespace {
const int kDivider = 16;
const std::array<std::array<int, 3>, 3> kKernel = {{{1, 2, 1}, {2, 4, 2}, {1, 2, 1}}};

uint8_t GetPixelMirror(const std::vector<uint8_t> &data, int col, int row, int width, int height) {
  int x = col;
  int y = row;

  if (x < 0) {
    x = -x - 1;
  } else if (x >= width) {
    x = (2 * width) - x - 1;
  }
  if (y < 0) {
    y = -y - 1;
  } else if (y >= height) {
    y = (2 * height) - y - 1;
  }
  return data[(static_cast<size_t>(y) * static_cast<size_t>(width)) + static_cast<size_t>(x)];
}
}  // namespace

PikhotskiyRVerticalGaussFilterSEQ::PikhotskiyRVerticalGaussFilterSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool PikhotskiyRVerticalGaussFilterSEQ::ValidationImpl() {
  const auto &in = GetInput();

  if (in.width <= 0 || in.height <= 0) {
    return false;
  }
  if (in.data.size() != static_cast<size_t>(in.width) * static_cast<size_t>(in.height)) {
    return false;
  }
  return true;
}

bool PikhotskiyRVerticalGaussFilterSEQ::PreProcessingImpl() {
  return true;
}

bool PikhotskiyRVerticalGaussFilterSEQ::RunImpl() {
  const auto &in = GetInput();

  int width = in.width;
  int height = in.height;
  const std::vector<uint8_t> &src = in.data;
  std::vector<uint8_t> dst(static_cast<size_t>(width) * static_cast<size_t>(height));

  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      int sum = 0;

      sum += kKernel[0][0] * GetPixelMirror(src, col - 1, row - 1, width, height);
      sum += kKernel[0][1] * GetPixelMirror(src, col, row - 1, width, height);
      sum += kKernel[0][2] * GetPixelMirror(src, col + 1, row - 1, width, height);

      sum += kKernel[1][0] * GetPixelMirror(src, col - 1, row, width, height);
      sum += kKernel[1][1] * GetPixelMirror(src, col, row, width, height);
      sum += kKernel[1][2] * GetPixelMirror(src, col + 1, row, width, height);

      sum += kKernel[2][0] * GetPixelMirror(src, col - 1, row + 1, width, height);
      sum += kKernel[2][1] * GetPixelMirror(src, col, row + 1, width, height);
      sum += kKernel[2][2] * GetPixelMirror(src, col + 1, row + 1, width, height);

      dst[(static_cast<size_t>(row) * static_cast<size_t>(width)) + static_cast<size_t>(col)] =
          static_cast<uint8_t>(sum / kDivider);
    }
  }

  GetOutput().width = width;
  GetOutput().height = height;
  GetOutput().data = std::move(dst);
  return true;
}

bool PikhotskiyRVerticalGaussFilterSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace pikhotskiy_r_vertical_gauss_filter
