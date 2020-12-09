from torchtext import data
import os
import random

class MR(data.Dataset):
    @classmethod
    def splits(cls, path, fields=None, test_rate=0.1, shuffle=True, **kwargs):
        examples = []
        with open(os.path.join(path, "rt-polarity.neg")) as f:
            examples += [data.Example.fromlist([line.strip(), 0], fields) for line in f]

        with open(os.path.join(path, "rt-polarity.pos")) as f:
            examples += [data.Example.fromlist([line.strip(), 1], fields) for line in f]

        test_size = int(len(examples) * test_rate)
        if shuffle: random.shuffle(examples)
        test_examples = examples[:test_size]
        train_examples = examples[test_size:]
        return cls(train_examples, fields, **kwargs), cls(test_examples, fields, **kwargs)
