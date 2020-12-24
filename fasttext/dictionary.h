#include <vector>
#include <iostream>
#include <string>
#include "assert.h"
#include <unordered_map>
#include "real.h"

namespace fasttext {
    typedef int32_t id_type;
    enum class entry_type: int32_t {word=0, label=1};

    struct Entry {
        std::string word;
        entry_type type;
        int64_t count;
        std::vector<int32_t> subword;
    };
    class Dictionary {
        private:
        static const int32_t MAX_VOCAB_SIZE = 30000000;
        static const int32_t MAX_LINE_SIZE = 1024;

        int32_t find(const std::string&) const;
        int32_t find(const std::string&, uint32_t h) const;
        std::shared_ptr<Args> args_;
        std::vector<int32_t> word2int_;
        std::vector<Entry> words_;

        std::vector<real> pdiscard_;
        int32_t size_;
        int32_t nwords_;
        int32_t nlabels_;
        int64_t ntokens_;

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
        inline int32_t nwords() const { return nwords_;}
        inline int32_t size() const { return size_; }
        inline int32_t nlabels() const { return nlabels_;}
        inline int64_t ntokens() const { return ntokens_;};

        void checkidx(int32_t id) const { assert(id >= 0); assert(id <= nwords_); }

        int32_t getId(const std::string& word) const { return getId(word, hash(word)); };
        int32_t getId(const std::string& word, uint32_t h) const {
            return word2int_[find(word, h)];
        };

        entry_type getType(int32_t id) const {assert(id >= 0); assert(id < size_); return words_[id].type;}; 
        entry_type getType(const std::string& word) const { return word.find(args_->label) == 0 ? entry_type::label : entry_type::word;};

        bool discard(int32_t, real) const;

        inline std::string getWord(int32_t id) const { checkidx(id); return words_[id].word;}

        const std::vector<int32_t>& getSubwords(int32_t id) const { checkidx(id); return words_[id].subword;};
        const std::vector<int32_t> getSubwords(const std::string& word) const {
            int32_t id = find(word);
            return getSubwords(id);
        };

        void getSubwords(
            const std::string&,
            std::vector<int32_t>&,
            std::vector<std::string>&) const;



        int32_t hash(const std::string& word) const;
        inline int32_t find(const std::string& word) {return find(word, hash(word));}

        int32_t find(const std::string& word, int32_t h) { 
            int32_t id = h % MAX_VOCAB_SIZE;
            while (word2int_[id] != -1 && words_[word2int_[id]].word!=word) {
                id = (id + 1) % MAX_VOCAB_SIZE;
            }
            return id;
        }

        void getSubwords(
            const std::string&,
            std::vector<int32_t>&, 
            std::vector<std::string>&) const;

        void computeSubwords(
            const std::string&, std::vector

        );

        void add(std::string word) {
            int32_t id = find(word);
            ntokens_ ++;
            if (word2int_[id] == -1) 
            {
                Entry e;
                e.word = word;
                e.type = getType(word);
                e.count = 1;
                words_.push_back(e);
                word2int_[id] = size_++; // 新添加的word 在 words 中的 idx 为 size_ ++
            }  
            else
            {
                words_[word2int_[id]].count++;
            }
            
        }


        

    };

};
