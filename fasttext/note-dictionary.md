
[toc]

# fasttext 源码解析 -- dictionary

## 头文件解析

```cpp
#pragma once

#include <istream>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "args.h"
#include "real.h"

namespace fasttext {

typedef int32_t id_type;
enum class entry_type : int8_t { word = 0, label = 1 };

struct entry {
  std::string word;
  int64_t count;
  entry_type type;
  std::vector<int32_t> subwords;
};

class Dictionary {
 protected:
  static const int32_t MAX_VOCAB_SIZE = 30000000;
  static const int32_t MAX_LINE_SIZE = 1024;

  int32_t find(const std::string&) const; // 线性探查 找到 word 在 word2int_ 中的 idx
  int32_t find(const std::string&, uint32_t h) const; // 线性探查 找到 word 在 word2int_ 中的 idx
  void initTableDiscard(); // 填充 pdiscard_ 向量
  void initNgrams(); // 初始化 Ngrams。先 add 词，再一次性 initNgrams
  void reset(std::istream&) const;
  void pushHash(std::vector<int32_t>&, int32_t) const;
  void addSubwords(std::vector<int32_t>& line, const std::string& word, int32_t wid) const; // 将根据 word 和 wid 获取 subwords 并添加到 line 中

  std::shared_ptr<Args> args_;
  std::vector<int32_t> word2int_;   // 这个是 hash 表 
                                    // hashid -> word 在 words_ 中对应的 idx
  std::vector<entry> words_;        // 词表

  std::vector<real> pdiscard_; // 每个单词对应的丢弃概率
  int32_t size_;
  int32_t nwords_; // 词表的大小
  int32_t nlabels_;
  int64_t ntokens_; // 文章中有多少个单词

  int64_t pruneidx_size_;
  std::unordered_map<int32_t, int32_t> pruneidx_;
  void addWordNgrams(
      std::vector<int32_t>& line,
      const std::vector<int32_t>& hashes,
      int32_t n) const;

 public:
  static const std::string EOS;
  static const std::string BOW;
  static const std::string EOW;

  explicit Dictionary(std::shared_ptr<Args>);
  explicit Dictionary(std::shared_ptr<Args>, std::istream&);
  int32_t nwords() const;
  int32_t nlabels() const;
  int64_t ntokens() const;

  int32_t getId(const std::string&) const;          // 返回 word 在 words_ 中的 idx
  int32_t getId(const std::string&, uint32_t h) const; // 根据 word 和 hash值 h 来获取在 words_ 中到 idx

  entry_type getType(int32_t) const;
  entry_type getType(const std::string&) const;

  bool discard(int32_t, real) const;

  std::string getWord(int32_t) const;
  const std::vector<int32_t>& getSubwords(int32_t wid) const; // 根据 wid 获取 subword，并返回
  const std::vector<int32_t> getSubwords(const std::string& word) const; // 根据 word 获取 subword，并返回
  void getSubwords( // 根据 word 获取 subword 然后添加到 ngrams 和 substrings 中
        const std::string& word,
        std::vector<int32_t>& ngrams,
        std::vector<std::string>& substrings) const {
  void computeSubwords( //  计算 word 的 ngram，并添加到 ngrams 和 substrings 中 
                        // ngrams 是 int32_t 数组 substrings 是 string 数组
      const std::string& word,
      std::vector<int32_t>& ngrams,
      std::vector<std::string>* substrings = nullptr) const;
  uint32_t hash(const std::string& str) const; // 对一个字符串进行 hash
  void add(const std::string& word); // 添加 word 到词典中
                                      //  注意，这个操作不会计算 subwords, subwords 是通过 initNgrams 来计算的
  bool readWord(std::istream& in, std::string& word) const; // 从 in 中读取一个词到 word
  void readFromFile(std::istream&);
  std::string getLabel(int32_t) const;
  void save(std::ostream&) const;
  void load(std::istream&);
  std::vector<int64_t> getCounts(entry_type type) const; // 获取 类型为 type 的单词的频率向量
  int32_t getLine(std::istream& in, std::vector<int32_t>& words, std::vector<int32_t>& labels)
      const;
  int32_t getLine(std::istream& in, std::vector<int32_t>& words, std::minstd_rand& rng)
      const; // 从 in 中获取一行，将其中的单词进行解析并添加到 words 中，在这个过程中使用 dischard 函数进行随机丢弃, 最后返回采样的单词数
  void threshold(int64_t t, int64_t tl);  // 去掉小于频数小于 t 的word，频数小于 tl 的 label
  void prune(std::vector<int32_t>&); // 清理压缩 hash 关系
  bool isPruned() {
    return pruneidx_size_ >= 0;
  }
  void dump(std::ostream&) const;
  void init();
};

} // namespace fasttext
```

### 1. 从输入数据构造词典的整体流程

```cpp
void Dictionary::readFromFile(std::istream& in) {
  std::string word;
  int64_t minThreshold = 1;
  // 1. 逐词读取
  while (readWord(in, word)) {
    // 2. 将词添加到词典中
    add(word);
    if (ntokens_ % 1000000 == 0 && args_->verbose > 1) {
      std::cerr << "\rRead " << ntokens_ / 1000000 << "M words" << std::flush;
    }
    // 如果超出词典容量的 3/4，则去除低频词
    if (size_ > 0.75 * MAX_VOCAB_SIZE) {
      minThreshold++;
      // 去除低频词
      threshold(minThreshold, minThreshold);
    }
  }
  // 去除低频词，并按照词频降序排序
  threshold(args_->minCount, args_->minCountLabel);
  initTableDiscard();
  // 基于n-gram，初始化sub-word
  initNgrams();
  if (args_->verbose > 0) {
    std::cerr << "\rRead " << ntokens_ / 1000000 << "M words" << std::endl;
    std::cerr << "Number of words:  " << nwords_ << std::endl;
    std::cerr << "Number of labels: " << nlabels_ << std::endl;
  }
  if (size_ == 0) {
    throw std::invalid_argument(
        "Empty vocabulary. Try a smaller -minCount value.");
  }
}
```

### 2. 如何从 in 中读取一个词到 word 中

为什么用 `streambuf`？ 为什么不直接读取？这样效率会高吗？

```cpp
// 1. 对于词向量训练，需要先分词，然后词之前用空格隔开
bool Dictionary::readWord(std::istream& in, std::string& word) const {
  int c;
  // 1. 获取文件流的指针
  std::streambuf& sb = *in.rdbuf();
  word.clear();
  // 2. 循环读取，每次从文件流中读取一个char
  while ((c = sb.sbumpc()) != EOF) {
    // 3. 对c读取的字符做不同的处理，如果不是空格等，则继续读取下一个字符
    if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' ||
        c == '\f' || c == '\0') {
      if (word.empty()) {
        if (c == '\n') { // 
          word += EOS;
          return true;
        }
        continue;
      } else {
        if (c == '\n')
          sb.sungetc();
        return true;
      }
    }
    // 4. 将char添加到word中，继续读取下一个字符
    word.push_back(c);
  }
  // trigger eofbit
  in.get();
  return !word.empty();
}
```

### 3. add 添加一个词到词典 words_ 中

```cpp
void Dictionary::add(const std::string& w) {
  // 1. 通过find获取词的hash值
  int32_t h = find(w);
  ntokens_++;
  // 2. 通过hash值，查询该词是否在表word2int_中。
  //    该表的下标为词的hash值，value为词的id，容量为 MAX_VOCAB_SIZE
  if (word2int_[h] == -1) {
    // 3. 新词，将其添加到词典 words_中
    entry e;
    e.word = w;
    e.count = 1; // 新词，词频为1
    e.type = getType(w); // 词的类型，分类则为label，词向量则为word，即将所有的词放在一个词典中的
                         // 没有分开存储label与word
    words_.push_back(e);
    word2int_[h] = size_++; // 添加词的id，id就是个顺序值，和普通的for循环中的i作为id是一样的
  } else {
    // 词典中已存在的词，仅增加词频
    words_[word2int_[h]].count++;
  }
}
```

### 4. 如何去低频词？

```cpp
void Dictionary::threshold(int64_t t, int64_t tl) {
  // 1. 先对词典中的词按照词频排序，
  sort(words_.begin(), words_.end(), [](const entry& e1, const entry& e2) {
    if (e1.type != e2.type) {
      return e1.type < e2.type;
    }
    // 词频降序排列
    return e1.count > e2.count;
  });
  // 2. 将 word 词频小于t的删除，将label词频小于t1的删除
  words_.erase(
      remove_if(
          words_.begin(),
          words_.end(),
          [&](const entry& e) {
            return (e.type == entry_type::word && e.count < t) ||
                (e.type == entry_type::label && e.count < tl);
          }),
      words_.end());
  // 3. 词典容量调整，前面删除了部分词。
  words_.shrink_to_fit();
  // 4. 重置词典数据
  size_ = 0;
  nwords_ = 0;
  nlabels_ = 0;
  std::fill(word2int_.begin(), word2int_.end(), -1);
  // 将词典中的数据重新计算id值
  for (auto it = words_.begin(); it != words_.end(); ++it) {
    int32_t h = find(it->word); // 找到 word 在 word2int_ 中的下标
    word2int_[h] = size_++;  // 将 word2int[h] 设置为 size_ 并 size_++
    // 根据 type 来计数 nwords_ 或者 nlabels_
    if (it->type == entry_type::word) {
      nwords_++;
    }
    if (it->type == entry_type::label) {
      nlabels_++;
    }
  }
}
```

### 5. initTableDiscard

每个单词的舍弃概率为

$$
p(w o r d)=\sqrt{\frac{t}{f(w o r d)}}+\frac{t}{f(w o r d)}, \quad
$$ 

其中 $f(w o r d)=\frac{\operatorname{count}(w o r d)}{n t o k e n s_{-}}$

```cpp
void Dictionary::initTableDiscard() {
  // 将 大小调整为词典大小
  pdiscard_.resize(size_);
  for (size_t i = 0; i < size_; i++) {
    // 计算概率，词频/词总数
    real f = real(words_[i].count) / real(ntokens_);
    pdiscard_[i] = std::sqrt(args_->t / f) + args_->t / f;
  }
}
```

### 6. initNgrams

一开始添加词的时候是不计算 ngram 的，添加完所有词之后统一计算 ngram

```cpp
void Dictionary::initNgrams() {
  for (size_t i = 0; i < size_; i++) {
    // 1. 从词典中获取一个词，并给该词加上"<"与">"，例如：北京 -> "<北京>"
    std::string word = BOW + words_[i].word + EOW;
    words_[i].subwords.clear();
    // 该词的子词列表，首先添加全词的id，全词也算一个子词
    words_[i].subwords.push_back(i);
    if (words_[i].word != EOS) {
      // 依据n-gram，计算子词
      computeSubwords(word, words_[i].subwords);
    }
  }
}
```

ngram 是什么？

假设 `args_->maxn=3`, `args_->minn=2`
where 对应的 ngram 是 where, wh, whe, her, ere, re

```cpp
void Dictionary::computeSubwords( // 根据 word 计算 ngram，并填充 ngrams
    const std::string& word, // 如 word = "<终南山>"
    std::vector<int32_t>& ngrams,
    std::vector<std::string>* substrings = nullptr) const {
  for (size_t i = 0; i < word.size(); i++) {
    std::string ngram;
    if ((word[i] & 0xC0) == 0x80) { // 如果是一个多字节字符的中间字节，就忽略
      continue;
    }
    for (size_t j = i, n = 1; j < word.size() && n <= args_->maxn; n++) {
      ngram.push_back(word[j++]);
      while (j < word.size() && (word[j] & 0xC0) == 0x80) { // 这一句相当于 push_back 进去了一个 utf8字符
        ngram.push_back(word[j++]);
      }
      if (n >= args_->minn && !(n == 1 && (i == 0 || j == word.size()))) {
        int32_t h = hash(ngram) % args_->bucket;
        pushHash(ngrams, h);
        if (substrings) {
          substrings->push_back(ngram);
        }
      }
    }
  }
}
```

`computeSubwords` 中需要注意的是 `if((word[i] & 0xC0) == 0x80)` 这句代码。0xC0是 `1100 0000` 的十六进制表示，与某个字符进行"位与"操作后，就是提取这个字符的前两个bit位的数字，如果结果为 `0x80`（1000 0000），即该if代码的意思是如果字符 `word[i]` 的起始两位是 `10` 的话。那怎么整体理解这几句代码呢？我的理解是，如果训练语料是全英文的，就没必要写这么复杂，但如果是其他语种的语料，这样写法就可以按照相应的utf-8编码来拆分该语种的每个词。

UTF-8是一种变长字节编码方式。对于某一个字符的UTF-8编码，如果只有一个字节则其最高二进制位为0；如果是多字节，其第一个字节从最高位开始，连续的二进制位值为1的个数决定了其编码的位数，其余各字节均以10开头。UTF-8最多可用到6个字节。

根据 UTF-8 编码的性质，如果某个字节以 `10` 开头，那么这个字节是某个多字节字符的中间字节。

### pushHash

```cpp
void Dictionary::pushHash(std::vector<int32_t>& hashes, int32_t id) const {
  if (pruneidx_size_ == 0 || id < 0) {
    return;
  }
  if (pruneidx_size_ > 0) {
    if (pruneidx_.count(id)) {
      id = pruneidx_.at(id);
    } else {
      return;
    }
  }
  hashes.push_back(nwords_ + id); // ngrams 的 id 在单词之后
}
```


# References
1. [【NLP】【七】fasttext源码解析 - muqiusangyang的个人空间 - OSCHINA - 中文开源技术交流社区](https://my.oschina.net/u/3800567/blog/2877570)
2. [fasttext源码解析（1） - 知乎](https://zhuanlan.zhihu.com/p/64960839)