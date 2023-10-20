#include <iostream>
#include "absl/status/status.h"
#include "tensorflow/lite/core/c/common.h"


absl::Status Preprocess(
    const std::vector<TfLiteTensor*>& input_tensors, const std::string& context,
    const std::string& query);