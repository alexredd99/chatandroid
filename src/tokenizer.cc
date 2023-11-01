#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <iostream>

#include "tokenizer.h"

template <typename C>
C concat(const C& lhs, const C& rhs) {
  C res(lhs.size() + rhs.size());
  typename C::iterator it = std::copy(lhs.cbegin(), lhs.cend(), res.begin());
  std::copy(rhs.cbegin(), rhs.cend(), it);
  return res;
}


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
    wordpiece_tokenizer_ = new WordpieceTokenizer(token2id_, "[UNK]", 512);
  }

  std::map<std::string, std::vector<int>> tokenize(std::string question, std::string context) {
    std::vector<std::string> context_ids_ = wordpiece_tokenizer_->tokenize(context);
    std::vector<std::string> question_ids_ = wordpiece_tokenizer_->tokenize(question);

    std::vector<int> input_ids;
    std::vector<int> position_ids;
    std::vector<int> attention_mask;

    //TODO: FIX THIS SHIT...
    input_tokens_.clear();
    input_tokens_.push_back(cls_token_);
    input_tokens_.insert(input_tokens_.end(), question_ids_.begin(), question_ids_.end());
    input_tokens_.push_back(sep_token_);
    input_tokens_.insert(input_tokens_.end(), context_ids_.begin(), context_ids_.end());
    input_tokens_.push_back(sep_token_);

    transfom_token2id(&input_tokens_, &input_ids);
    std::vector<int> question_pos_ids(question_ids_.size() + 2, 0);
    std::vector<int> context_pos_ids(context_ids_.size() + 1, 1);

    position_ids.clear();
    position_ids.insert(position_ids.end(), question_pos_ids.begin(), question_pos_ids.end());
    position_ids.insert(position_ids.end(), context_pos_ids.begin(), context_pos_ids.end());


    attention_mask = std::vector<int>(input_ids.size(), 1);

    std::map<std::string, std::vector<int>> encoding;
    encoding["input_ids"] = input_ids;
    encoding["position_ids"] = position_ids;
    encoding["attention_mask"] = attention_mask;

    return encoding;
  }

  //TODO: Use capitalize_next bool for proper capitalization
  //TODO: Remove output not in clause
  std::string decode(unsigned int start_idx, unsigned int end_idx) {
    std::string output("");

    for (unsigned int i = start_idx; i <= end_idx; i++) {
      std::string token = input_tokens_[i];

      if (token.compare(0, 2, "##") == 0) {
        token = token.substr(2, token.length() - 2);
      } else if ((token == cls_token_) || (token == sep_token_)) {
        continue;
      } else if (!ispunct(token.c_str()[0])) {
        if (i > start_idx) {
          token = " " + token;
        } else {
          token[0] = toupper(token[0]);
        }
      }

      output.append(token);
    }

    std::cout << std::endl;

    return output;
  }

  std::vector<std::string> input_tokens_;
private:
  WordpieceTokenizer* wordpiece_tokenizer_;
  std::map<std::string, int> token2id_; // vocab
  std::map<int, std::string> id2token_; // decode

  std::string cls_token_ = "[CLS]";
  std::string sep_token_ = "[SEP]";

  void transfom_token2id(std::vector<std::string>* input, std::vector<int>* output) {
    for (auto& token : *input) {
      output->push_back(token2id_.at(token));
    }
  }
};

// Pointer to Implementation Pattern

Tokenizer::Tokenizer(std::string vocab_path) {
  bertTokenizer = new MobileBertTokenizer(vocab_path);
}

std::map<std::string, std::vector<int>> Tokenizer::tokenize(std::string question, std::string context) {
  return bertTokenizer->tokenize(question, context);
}

std::string Tokenizer::decode(unsigned int start_idx, unsigned int end_idx) {
  return bertTokenizer->decode(start_idx, end_idx);
}
