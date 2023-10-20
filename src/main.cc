#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_plugin.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/tools/evaluation/utils.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/nnapi/NeuralNetworksTypes.h"
#include "tensorflow/lite/tools/logging.h"
#include "tensorflow/lite/core/c/common.h"
using tflite::StatefulNnApiDelegate;

int main(void){
  StatefulNnApiDelegate::Options options;
  options.accelerator_name = "google-edgetpu";
  options.use_burst_computation = true;
  options.disallow_nnapi_cpu = true;
  options.execution_preference = StatefulNnApiDelegate::Options::kSustainedSpeed;
  options.execution_priority = ANEURALNETWORKS_PRIORITY_HIGH;
  
  auto delegate = new StatefulNnApiDelegate(options);
  return 0;
}