#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

std::map<std::string, std::vector<int>> tokenize_test(std::string vocab_path, std::string question, std::string context);
std::string decode_test(unsigned int start_idx, unsigned int end_idx);