import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        s = set()
        for sentence in positive + negative:
            s.update(sentence.split(" "))
        vocab = sorted(s)

        dic = {}
        for i in range(len(vocab)):
            dic[vocab[i]] = i + 1
        
        res = []
        for sentence in positive + negative:
            cur = []
            words = sentence.split(" ")
            for word in words:
                cur.append(dic[word])
            res.append(cur)

        tokens = []
        for sentence in res :
            tokens.append(torch.tensor(sentence))
        tokens = torch.nn.utils.rnn.pad_sequence(tokens, padding_value = 0, batch_first = True)

        return tokens
        

        
