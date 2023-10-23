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

#include "tokenizer.h"

using tflite::StatefulNnApiDelegate;

#define MODEL_PATH "/home/alexander/workspace/chatandroid/data/lite-model_edgetpu_nlp_mobilebert-edgetpu_xs_1.tflite"

unsigned int arg_max(float* input, size_t size){
  unsigned int max = 0;
  for (unsigned int i = 0; i < size; i++){
    if(input[i] > input[max]){
      max = i;
    }
  }
  return max;
}

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

  //std::vector<TfLiteTensor*> inputs = {
  //  interpreter->input_tensor(0), // input ids
  //  interpreter->input_tensor(1), // segment/type ids
  //  interpreter->input_tensor(2)  // attention mask
  //};

  std::string context = "Machine learning (ML) is the scientific study of algorithms and statistical models that computer systems use to progressively improve their performance \
                on a specific task. Machine learning algorithms build a mathematical model of sample data, known as \"training data\", in order to make predictions or \
                decisions without being explicitly programmed to perform the task. Machine learning algorithms are used in the applications of email filtering, detection \
                of network intruders, and computer vision, where it is infeasible to develop an algorithm of specific instructions for performing the task.";

  std::string question = "What is Machine Learning?";

  std::string vocab_path("/home/alexander/workspace/chatandroid/data/vocab.txt");

  //WordpieceTokenizer tokenizer(vocab, "[UNK]", 512);
  std::map<std::string, std::vector<int>> encoded;
  encoded = tokenize_test(vocab_path, question, context);
  
  int* input_ids      = interpreter->typed_input_tensor<int>(0);
  int* type_ids       = interpreter->typed_input_tensor<int>(1);
  int* attention_mask = interpreter->typed_input_tensor<int>(2);
  
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

  float* end_logits   = interpreter->typed_output_tensor<float>(0);
  float* start_logits = interpreter->typed_output_tensor<float>(1);

  unsigned int end_idx = arg_max(end_logits, interpreter->output_tensor(0)->dims->data[1]);
  unsigned int start_idx = arg_max(start_logits, interpreter->output_tensor(1)->dims->data[1]);

  std::string output = decode_test(start_idx, end_idx);
  std::cout << output << std::endl;

  return 0;
}