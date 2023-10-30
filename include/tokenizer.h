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
  std::string decode(unsigned int start_idx, unsigned int end_idx);

private:
  MobileBertTokenizer* bertTokenizer;
};