#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/nnapi/NeuralNetworksTypes.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"

#include "tokenizer.h"

#include <iostream>

//#define DEBUG 1

void print_array(int* arr, unsigned int len, std::string name) {
  std::cout << name << std::endl;
  for (unsigned int i = 0; i < len; i++) {
    std::cout << arr[i] << ", ";
  }
  std::cout << std::endl;
}

void print_array(float* arr, unsigned int len, std::string name) {
  std::cout << name << std::endl;
  for (unsigned int i = 0; i < len; i++) {
    std::cout << arr[i] << ", ";
  }
  std::cout << std::endl;
}

unsigned int arg_max(float* input, size_t size) {
  unsigned int max = 0;
  for (unsigned int i = 0; i < size; i++) {
    if (input[i] > input[max]) {
      max = i;
    }
  }
  return max;
}

std::string get_text(void) {
  std::string text("");
  std::string line("");

  while (std::getline(std::cin, line)) {
    if (line.empty()) {
      break;
    }
    text.append(line);
  }

  return text;
}

ABSL_FLAG(std::string, model_path, "", "path to tflite model");
ABSL_FLAG(std::string, vocab_path, "", "path to vocab file");
ABSL_FLAG(std::string, device, "cpu", "device for TFLITE delegate");

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  std::string model_path = absl::GetFlag(FLAGS_model_path);
  std::string vocab_path = absl::GetFlag(FLAGS_vocab_path);

  if (model_path.length() == 0) {
    throw std::invalid_argument("Didn't to set model path");
  }

  if (vocab_path.length() == 0) {
    throw std::invalid_argument("Didn't to set vocab path");
  }

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  if (!model) {
    throw std::invalid_argument("Failed to open model");
  }

  std::string device = absl::GetFlag(FLAGS_device);

  TfLiteDelegate* delegate = nullptr;

  if (device == "tpu") {
    tflite::StatefulNnApiDelegate::Options options;
    options.accelerator_name = "google-edgetpu";
    options.use_burst_computation = true;
    options.disallow_nnapi_cpu = true;
    options.execution_preference = tflite::StatefulNnApiDelegate::Options::kSustainedSpeed;
    options.execution_priority = ANEURALNETWORKS_PRIORITY_HIGH;

    delegate = new tflite::StatefulNnApiDelegate(options);
  } else {
    TfLiteXNNPackDelegateOptions xnnOptions = TfLiteXNNPackDelegateOptionsDefault();
    // Change options...
    delegate = TfLiteXNNPackDelegateCreate(&xnnOptions);
  }

  if (!delegate) {
    throw std::invalid_argument("Bad " + device + " delegate options");
  }

  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);

  if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
    throw std::invalid_argument("Failed to apply delegate");
  };

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    throw std::runtime_error("Failed to allocate tensors");
  }


  std::cout << "Using model path: " << model_path << std::endl;
  std::cout << "Using vocab path: " << vocab_path << std::endl;
  std::cout << "Running on " << device << std::endl;

  //tflite::PrintInterpreterState(interpreter.get());

  //std::vector<TfLiteTensor*> inputs = {
  //  interpreter->input_tensor(0), // input ids
  //  interpreter->input_tensor(1), // segment/type ids
  //  interpreter->input_tensor(2)  // attention mask
  //};

  auto tokenizer = Tokenizer(vocab_path);

  int* input_ids = interpreter->typed_input_tensor<int>(0);
  int* type_ids = interpreter->typed_input_tensor<int>(1);
  int* attention_mask = interpreter->typed_input_tensor<int>(2);

  std::cout << "Input context, press enter twice when done." << std::endl << std::endl;
  std::cout << "You can continue asking questions on the same context." << std::endl;

  std::cout << "Context:" << std::endl;
  std::string context = get_text();

  while (true) {
    std::cout << "Question:" << std::endl;
    std::string question = get_text();

    if (question.empty()) { // detect control d?
      return 0;
    }

    std::map<std::string, std::vector<int>> encoded;

    encoded = tokenizer.tokenize(question, context);

    std::copy(encoded["input_ids"].begin(),
      encoded["input_ids"].end(),
      input_ids);
    std::copy(encoded["position_ids"].begin(),
      encoded["position_ids"].end(),
      type_ids);
    std::copy(encoded["attention_mask"].begin(),
      encoded["attention_mask"].end(),
      attention_mask);

    interpreter->Invoke();

    float* end_logits = interpreter->typed_output_tensor<float>(0);
    float* start_logits = interpreter->typed_output_tensor<float>(1);

    unsigned int end_idx = arg_max(end_logits, interpreter->output_tensor(0)->dims->data[1]);
    unsigned int start_idx = arg_max(start_logits, interpreter->output_tensor(1)->dims->data[1]);

#ifdef DEBUG
    print_array(input_ids, interpreter->input_tensor(0)->dims->data[1], "input_ids");
    print_array(type_ids, interpreter->input_tensor(1)->dims->data[1], "type_ids");
    print_array(attention_mask, interpreter->input_tensor(2)->dims->data[1], "attention_mask");

    std::cout << "end_idx=" << end_idx << std::endl;
    std::cout << "start_idx=" << start_idx << std::endl;

    print_array(input_ids + start_idx, end_idx - start_idx + 1, "pred");
#endif

    std::string output = tokenizer.decode(start_idx, end_idx);
    std::cout << "Answer: " << std::endl << output << std::endl << std::endl;
  }

  return 0;
}