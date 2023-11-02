#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

// Pointer to Implementation Pattern
class MobileBertTokenizer;

class Tokenizer {
public:
  Tokenizer(std::string vocab_path);
  std::map<std::string, std::vector<int>> tokenize(std::string question, std::string context);
  std::string decode(float* start_logits, size_t start_len, float* end_logits, size_t end_len);

private:
  MobileBertTokenizer* bertTokenizer;
};