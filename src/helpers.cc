#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/tools/evaluation/utils.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/nnapi/NeuralNetworksTypes.h"

#include <iostream>
#include "absl/status/status.h"
#include "absl/strings/str_split.h"
#include "absl/container/flat_hash_map.h"


// Maps index of input token to index of untokenized word from original input.
absl::flat_hash_map<size_t, size_t> token_to_orig_map;
// Original tokens of context.
std::vector<std::string> orig_tokens_;

struct TokenizerResult {
  std::vector<std::string> subwords;
};


absl::Status Preprocess(
    const std::vector<TfLiteTensor*>& input_tensors, const std::string& context,
    const std::string& query) {
  
  TfLiteTensor* ids_tensor = input_tensors[0];
  TfLiteTensor* segment_ids_tensor = input_tensors[1];
  TfLiteTensor* mask_tensor = input_tensors[2];


  token_to_orig_map.clear();
  
  // The orig_tokens is used for recovering the answer string from the index,
  // while the processed_tokens is lower-cased and used to generate input of
  // the model.
  orig_tokens_ = absl::StrSplit(context, absl::ByChar(' '), absl::SkipEmpty());
  std::vector<std::string> processed_tokens(orig_tokens_);

  std::string processed_query = query;
  // convert to lower case
  for (auto& token : processed_tokens) {
    absl::AsciiStrToLower(&token);
  }
  absl::AsciiStrToLower(&processed_query);


  TokenizerResult query_tokenize_results;
  //query_tokenize_results = tokenizer_->Tokenize(processed_query);

  for(std::string i : processed_tokens){
    std::cout << i << std::endl;
  }

/*
  std::vector<std::string> query_tokens = query_tokenize_results.subwords;
  if (query_tokens.size() > kMaxQueryLen) {
    query_tokens.resize(kMaxQueryLen);
  }

  // Example:
  // context:             tokenize     me  please
  // all_doc_tokens:      token ##ize  me  plea ##se
  // token_to_orig_index: [0,   0,     1,  2,   2]

  std::vector<std::string> all_doc_tokens;
  std::vector<int> token_to_orig_index;
  for (size_t i = 0; i < processed_tokens.size(); i++) {
    const std::string& token = processed_tokens[i];
    std::vector<std::string> sub_tokens = tokenizer_->Tokenize(token).subwords;
    for (const std::string& sub_token : sub_tokens) {
      token_to_orig_index.emplace_back(i);
      all_doc_tokens.emplace_back(sub_token);
    }
  }

  // -3 accounts for [CLS], [SEP] and [SEP].
  int max_context_len = kMaxSeqLen - query_tokens.size() - 3;
  if (all_doc_tokens.size() > max_context_len) {
    all_doc_tokens.resize(max_context_len);
  }

  std::vector<std::string> tokens;
  tokens.reserve(3 + query_tokens.size() + all_doc_tokens.size());
  std::vector<int> segment_ids;
  segment_ids.reserve(kMaxSeqLen);

  // Start of generating the features.
  tokens.emplace_back("[CLS]");
  segment_ids.emplace_back(0);

  // For query input.
  for (const auto& query_token : query_tokens) {
    tokens.emplace_back(query_token);
    segment_ids.emplace_back(0);
  }

  // For Separation.
  tokens.emplace_back("[SEP]");
  segment_ids.emplace_back(0);

  // For Text Input.
  for (int i = 0; i < all_doc_tokens.size(); i++) {
    auto& doc_token = all_doc_tokens[i];
    tokens.emplace_back(doc_token);
    segment_ids.emplace_back(1);
    token_to_orig_map_[tokens.size()] = token_to_orig_index[i];
  }

  // For ending mark.
  tokens.emplace_back("[SEP]");
  segment_ids.emplace_back(1);

  std::vector<int> input_ids(tokens.size());
  input_ids.reserve(kMaxSeqLen);
  // Convert tokens back into ids
  for (int i = 0; i < tokens.size(); i++) {
    auto& token = tokens[i];
    tokenizer_->LookupId(token, &input_ids[i]);
  }

  std::vector<int> input_mask;
  input_mask.reserve(kMaxSeqLen);
  input_mask.insert(input_mask.end(), tokens.size(), 1);

  int zeros_to_pad = kMaxSeqLen - input_ids.size();
  input_ids.insert(input_ids.end(), zeros_to_pad, 0);
  input_mask.insert(input_mask.end(), zeros_to_pad, 0);
  segment_ids.insert(segment_ids.end(), zeros_to_pad, 0);

  // input_ids INT32[1, 384]
  RETURN_IF_ERROR(PopulateTensor(input_ids, ids_tensor));
  // input_mask INT32[1, 384]
  RETURN_IF_ERROR(PopulateTensor(input_mask, mask_tensor));
  // segment_ids INT32[1, 384]
  RETURN_IF_ERROR(PopulateTensor(segment_ids, segment_ids_tensor));
  */
  return absl::OkStatus();
}