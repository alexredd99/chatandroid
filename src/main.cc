#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/nnapi/NeuralNetworksTypes.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/tools/evaluation/utils.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include <iostream>
#include "absl/status/status.h"
#include "absl/strings/str_split.h"

#include "helpers.h"

using tflite::StatefulNnApiDelegate;

#define MODEL_PATH "/home/alexander/workspace/chatandroid/data/lite-model_edgetpu_nlp_mobilebert-edgetpu_xs_1.tflite"

int main(void){
  //StatefulNnApiDelegate::Options options;
  //options.accelerator_name = "google-edgetpu";
  //options.use_burst_computation = true;
  //options.disallow_nnapi_cpu = true;
  //options.execution_preference = StatefulNnApiDelegate::Options::kSustainedSpeed;
  //options.execution_priority = ANEURALNETWORKS_PRIORITY_HIGH;
  //
  //auto delegate = new StatefulNnApiDelegate(options);
  
  auto model = tflite::FlatBufferModel::BuildFromFile(MODEL_PATH);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Failed to allocate tensors." << std::endl;
    return -2;
  }
  
  //tflite::PrintInterpreterState(interpreter.get());

  std::vector<TfLiteTensor*> inputs = {
    interpreter->input_tensor(0), // input ids
    interpreter->input_tensor(1), // segment/type ids
    interpreter->input_tensor(2)  // attention mask
  };

  std::string context = "my name is jim.";
  std::string question = "what is your name?";
  auto status = Preprocess(inputs, context, question);

  return 0;
}