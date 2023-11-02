#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iterator>

#include "tokenizer.h"
#include "helpers.h"


static void load_vocab(std::string vocab_path, std::map<std::string, int>* vocab) {
  std::fstream vocab_stream(vocab_path, std::ios::in);
  vocab->clear();

  std::string line;
  int n = 0;
  while (std::getline(vocab_stream, line)) {
    vocab->insert({ line, n });
    n++;
  }
}

static void load_vocab(std::string vocab_path, std::map<std::string, int>* vocab, std::map<int, std::string>* decode) {
  std::fstream vocab_stream(vocab_path, std::ios::in);
  vocab->clear();
  decode->clear();

  std::string line;
  int n = 0;

  while (std::getline(vocab_stream, line)) {
    vocab->insert({ line, n });
    decode->insert({ n, line });
    n++;
  }
}

std::vector<std::string> split_on_punc(std::string input) {
  std::vector<std::string> tokens;
  std::string sChar;

  bool new_token = true;

  for (unsigned int i = 0; i < input.length(); i++) {
    sChar = input.substr(i, 1);

    if (ispunct(sChar.c_str()[0])) {
      new_token = true;
      tokens.push_back(sChar);
      //tokens += " " + sChar + " ";
    } else {
      if (new_token) {
        new_token = false;
        tokens.push_back("");
      }
      tokens.back() += sChar;
      //tokens += sChar;
    }
  }

  return tokens;
}

std::vector<std::string> whitespace_tokenizer(std::string input) {
  std::vector<std::string> tokens;
  std::stringstream input_stream(input);
  std::string token;

  while (getline(input_stream, token, ' ')) {
    tokens.push_back(token);
  }

  return tokens;
}

size_t nested_vector_size(std::vector<std::vector<std::string>> input) {
  size_t total_size = 0;
  for (auto& v : input) {
    total_size += v.size();
  }
  return total_size;
}

class WordpieceTokenizer {
public:
  WordpieceTokenizer(std::string vocab_path, std::string unk_token, int max_input_chars_per_word) {
    load_vocab(vocab_path, &token2id);
    unk_token_ = unk_token;
    max_input_chars_per_word_ = max_input_chars_per_word;
  }

  WordpieceTokenizer(std::map<std::string, int> vocab, std::string unk_token, int max_input_chars_per_word) {
    token2id = vocab;
    unk_token_ = unk_token;
    max_input_chars_per_word_ = max_input_chars_per_word;
  }

  std::vector<std::string> tokenize(std::string input) {
    std::string lower_input = input;
    // put text into lower case
    std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);

    lower_input = split_on_punc(lower_input);

    std::vector<std::string> output_tokens;

    for (auto& token : whitespace_tokenizer(lower_input)) {
      if (token.length() > max_input_chars_per_word_) {
        output_tokens.push_back(unk_token_); // too long
        continue;
      }

      bool is_bad = false;
      unsigned int start = 0;
      std::vector<std::string> sub_tokens;

      while (start < token.length()) {
        unsigned int end = token.length();
        std::string cur_substr;
        while (start < end) {
          std::string substr = token.substr(start, end - start);

          if (start > 0) {
            // Don't add ## to punctuation
            if (!((substr.length() == 1) && ispunct(substr.c_str()[0]))) {
              substr = "##" + substr;
            }
          }

          if (token2id.count(substr)) {
            cur_substr = substr;
            break;
          }
          end -= 1;
        }

        if (cur_substr == "") {
          is_bad = true;
          break;
        }

        sub_tokens.push_back(cur_substr);
        start = end;
      }

      if (is_bad) {
        output_tokens.push_back(unk_token_);
      } else {
        output_tokens = concat(output_tokens, sub_tokens);
      }
    }
    return output_tokens;
  }

  std::map<std::string, int> token2id; // vocab

private:
  std::string unk_token_;
  unsigned int max_input_chars_per_word_;

  std::vector<std::string> whitespace_tokenizer(std::string input) {
    std::vector<std::string> tokens;
    std::stringstream input_stream(input);
    std::string token;

    while (getline(input_stream, token, ' ')) {
      tokens.push_back(token);
    }

    return tokens;
  }

  std::string split_on_punc(std::string input) {
    std::string tokens;
    std::string sChar;

    for (unsigned int i = 0; i < input.length(); i++) {
      sChar = input.substr(i, 1);

      if (ispunct(sChar.c_str()[0])) {
        tokens += " " + sChar + " ";
      } else {
        tokens += sChar;
      }
    }

    return tokens;
  }
};

class MobileBertTokenizer {
public:
  MobileBertTokenizer(std::string vocab_path) {
    load_vocab(vocab_path, &token2id_, &id2token_);
  }

  std::map<std::string, std::vector<int>> tokenize(std::string question, std::string context) {
    std::vector<std::string> split_question = whitespace_tokenizer(question);
    std::vector<std::string> split_context = whitespace_tokenizer(context);

    input_whitespace_tokenized_ = concat(split_question, split_context);

    std::vector<std::vector<std::string>> tokenized_question = wordpiece_tokenize(split_question);
    std::vector<std::vector<std::string>> tokenized_context = wordpiece_tokenize(split_context);

    // Add special tokens
    tokenized_question[0].insert(tokenized_question[0].begin(), cls_token_);
    tokenized_question.back().push_back(sep_token_);
    tokenized_context.back().push_back(sep_token_);

    auto input_tokens = concat(tokenized_question, tokenized_context);

    input_tokens_.clear();

    // For each input word
    for (unsigned int i = 0; i < input_tokens.size(); i++) {
      std::vector<std::string> curr_wordpiece = input_tokens[i];
      for (unsigned int j = 0; j < curr_wordpiece.size(); j++) {
        bert_id_idx2input_string_idx[input_tokens_.size()] = i;
        input_tokens_.push_back(curr_wordpiece[j]);
      }
    }

    std::vector<int> input_ids;
    std::vector<int> position_ids;
    std::vector<int> attention_mask;

    transfom_token2id(&input_tokens_, &input_ids);

    std::vector<int> question_pos_ids(nested_vector_size(tokenized_question), 0);
    std::vector<int> context_pos_ids(nested_vector_size(tokenized_context), 1);
    position_ids = concat(question_pos_ids, context_pos_ids);

    attention_mask = std::vector<int>(input_ids.size(), 1);

    std::map<std::string, std::vector<int>> encoding;
    encoding["input_ids"] = input_ids;
    encoding["position_ids"] = position_ids;
    encoding["attention_mask"] = attention_mask;

    return encoding;
  }

  //TODO: Use capitalize_next bool for proper capitalization
  //TODO: Remove output not in clause
  std::string decode(float* start_logits, size_t start_len, float* end_logits, size_t end_len) {
    unsigned int start_top5_idxs[5], end_top5_idxs[5] = { 0 };

    arg_maxN(start_logits, start_len, start_top5_idxs, 5);
    arg_maxN(end_logits, end_len, end_top5_idxs, 5);

    unsigned int bert_start_idx = get_first_valid_idx(start_top5_idxs, 5);
    unsigned int bert_end_idx = get_first_valid_idx(end_top5_idxs, 5);

    unsigned int input_string_start_idx = bert_id_idx2input_string_idx[bert_start_idx];
    unsigned int input_string_end_idx = bert_id_idx2input_string_idx[bert_end_idx];

    //std::cout << "bert_start_idx: " << bert_start_idx << std::endl;
    //std::cout << "bert_end_idx: " << bert_end_idx << std::endl;
    //std::cout << "input_string_start_idx: " << input_string_start_idx << std::endl;
    //std::cout << "input_string_end_idx: " << input_string_end_idx << std::endl;
    //std::cout << "input size: " << input_whitespace_tokenized_.size() << std::endl;

    std::vector<std::string> output_whitespace_tokenized =
      subset(input_whitespace_tokenized_, input_string_start_idx, input_string_end_idx);

    std::string output("");

    for (auto& str : output_whitespace_tokenized) {
      output += str + " ";
    }
    
    output[0] = std::toupper(output[0]);

    return output;
  }

  std::vector<std::string> input_whitespace_tokenized_;
  std::vector<std::string> input_tokens_;

private:
  std::map<std::string, int> token2id_; // vocab
  std::map<int, std::string> id2token_; // decode

  //TODO: Get this from vocab.txt or tokenizer.json
  std::string cls_token_ = "[CLS]";
  std::string sep_token_ = "[SEP]";
  std::string unk_token_ = "[UNK]";

  unsigned int max_input_chars_per_word_ = 512;

  unsigned int context_start_;
  std::map<unsigned int, unsigned int> bert_id_idx2input_string_idx;

  // Pass output from whitespace tokenizer
  std::vector<std::vector<std::string>> wordpiece_tokenize(std::vector<std::string> input) {
    std::vector<std::vector<std::string>> output;

    for (auto& word : input) {
      std::vector<std::string> sub_tokens;

      std::string lower_word = word;
      // Put word into lowercase
      std::transform(lower_word.begin(), lower_word.end(), lower_word.begin(), ::tolower);
      // Split on punctuation

      // If any subtokens are bad, just set as [UNK]
      for (auto& token : split_on_punc(lower_word)) {
        if (token.length() > max_input_chars_per_word_) {
          sub_tokens = { unk_token_ }; // too long
          //sub_tokens.push_back(unk_token_); // too long
          continue;
        }

        bool is_bad = false;
        unsigned int start = 0;

        while (start < token.length()) {
          unsigned int end = token.length();
          std::string cur_substr;

          while (start < end) {
            std::string substr = token.substr(start, end - start);

            if (start > 0) {
              // Don't add ## to punctuation?
              if (!((substr.length() == 1) && ispunct(substr.c_str()[0]))) {
                substr = "##" + substr;
              }
            }

            if (token2id_.count(substr)) {
              cur_substr = substr;
              break;
            }
            end -= 1;
          }

          if (cur_substr == "") {
            is_bad = true;
            break;
          }

          sub_tokens.push_back(cur_substr);
          start = end;
        }

        if (is_bad) {
          sub_tokens = { unk_token_ };
          break;
          //output_tokens.push_back(unk_token_);
        }
        // push substr back to sub tokens?
      }
      output.push_back(sub_tokens);
    }
    return output;
  }

  void transfom_token2id(std::vector<std::string>* input, std::vector<int>* output) {
    for (auto& token : *input) {
      output->push_back(token2id_.at(token));
    }
  }

  bool valid_output_idx(unsigned int idx) {
    //TODO: Don't hardcode, bad tokens from vocab in future
    if (idx < context_start_) {
      return false;
      //TODO: really jank, need to figure out how to store input tokens better..
    } else if (token2id_[input_tokens_[idx]] < 998) {// Unused and special characters
      return false;
    } else {
      return true;
    }
  }

  unsigned int get_first_valid_idx(unsigned int* idxs, size_t idxs_len) {
    for (unsigned int i = 0; i < idxs_len; i++) {
      //TODO: really jank, need to figure out how to store input tokens better..
      if (valid_output_idx(idxs[i])) {
        return idxs[i];
      }
    }

    return idxs[0]; //TODO: Should do proper error handling if none found
  }
};

// Pointer to Implementation Pattern

Tokenizer::Tokenizer(std::string vocab_path) {
  bertTokenizer = new MobileBertTokenizer(vocab_path);
}

std::map<std::string, std::vector<int>> Tokenizer::tokenize(std::string question, std::string context) {
  return bertTokenizer->tokenize(question, context);
}

std::string Tokenizer::decode(float* start_logits, size_t start_len, float* end_logits, size_t end_len) {
  return bertTokenizer->decode(start_logits, start_len, end_logits, end_len);
}
