#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, q, k, v, scale=None, attn_mask=None):
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention *= scale
        if attn_mask is not None:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = F.softmax(attention, dim=2)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context


# In[41]:


class MultiHeadAttention(torch.nn.Module):
    
    def __init__(self, embed_dim, n_heads, dropout_rate=0.0):
        super(MultiHeadAttention, self).__init__()
        self.dk = embed_dim // n_heads
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_qs = nn.ModuleList([nn.Linear(embed_dim, self.dk) for _ in range(n_heads)])
        self.linear_ks = nn.ModuleList([nn.Linear(embed_dim, self.dk) for _ in range(n_heads)])
        self.linear_vs = nn.ModuleList([nn.Linear(embed_dim, self.dk) for _ in range(n_heads)])
        self.scaled_dot_product_attention = ScaledDotProductAttention(dropout_rate)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.linear_final = nn.Linear(self.dk * n_heads, self.dk * n_heads)
        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant(m.weight, 1.0)
                nn.init.constant(m.bias, 0)

    def forward(self, Q, K, V, attn_mask=None):

        heads = []
        scale = self.dk ** (-0.5)
        for i in range(self.n_heads):
            q = self.linear_qs[i](Q)
            k = self.linear_ks[i](K)
            v = self.linear_vs[i](V)
            heads.append(self.scaled_dot_product_attention(q, k, v, scale, attn_mask))
        z = torch.cat(heads, dim=-1)

        residual = self.linear_final(z)
        output = self.layer_norm(Q + residual)
        return output

class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embed_dim):
        super(PositionalEmbedding, self).__init__()
        self.embedding_table = self.get_positional_table(max_seq_len, embed_dim)

    def get_positional_table(self, max_seq_len, embed_dim):

        def get_item(x, y):
            i = y // 2
            if y % 2 == 0:
                return np.sin(x / 10000 ** (2 * i / embed_dim))
            else:
                return np.cos(x / 10000 ** (2 * i / embed_dim))

        embedding_table = torch.tensor([[get_item(x, y) for y in range(embed_dim)] for x in range(max_seq_len)])
        return embedding_table

    def forward(self, x):
        batch_size = x.shape[0]
        return x + self.embedding_table.unsqueeze(0).expand(batch_size, -1, -1)

class FFN(nn.Module):

    def __init__(self, d_in, d_mid):
        super(FFN, self).__init__()
        d_out = d_in
        self.ffn = nn.Sequential(
            nn.Linear(d_in, d_mid),
            nn.ReLU(),
            nn.Linear(d_mid, d_out),
        )
        self.layer_norm = nn.LayerNorm(d_out)

    def forward(self, x):
        residual = self.ffn(x)
        return self.layer_norm(residual + x)


class EncoderLayer(nn.Module):

    def __init__(self, embed_dim, ffn_dim, n_heads, dropout_rate=0.0):
        super(EncoderLayer, self).__init__()
        self.multihead = MultiHeadAttention(embed_dim, n_heads, dropout_rate)
        self.ffn = FFN(embed_dim, ffn_dim)

    def forward(self, x, attn_mask=None):
        q, k, v = x, x, x
        x = self.multihead(q, k, v, attn_mask)
        return self.ffn(x)

class Encoder(nn.Module):

    def __init__(self, vocab_size, embed_dim, max_seq_len, n_heads, dropout_rate=0.0, n_layers=6):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder_layers = nn.ModuleList([EncoderLayer(embed_dim, embed_dim // 2, n_heads, dropout_rate) for _ in range(n_layers)])
        self.position_embedding = PositionalEmbedding(max_seq_len, embed_dim)

    def forward(self, enc_x):
        """

        :param enc_x:  shape=(batch_size, seq_len)
        :return:
        """

        embedded = self.embedding(enc_x)
        embedded += self.position_embedding(embedded)
        attn_mask = padding_mask(enc_x)
        output = embedded
        for encoder_layer in self.encoder_layers:
            output = encoder_layer(output, attn_mask)
        return output

def padding_mask(x):
    """

    :param x: shape=(batch_size, seq_len)
    :return:
    """

    seq_len = x.shape[1]
    mask = x.eq(0).unsqueeze(1).expand(-1, seq_len, -1)
    return mask

class DecoderLayer(nn.Module):

    def __init__(self, embed_dim, n_heads, dropout_rate=0.0):

        super(DecoderLayer, self).__init__()
        self.dec_self_attention = MultiHeadAttention(embed_dim, n_heads, dropout_rate)
        self.dec_enc_attention = MultiHeadAttention(embed_dim, n_heads, dropout_rate)
        self.ffn = FFN(embed_dim, embed_dim // 2)


    def forward(self, enc_outputs, dec_inputs):
        dec_inputs = self.dec_self_attention(dec_inputs, dec_inputs, dec_inputs)
        outputs = self.dec_enc_attention(dec_inputs, enc_outputs, enc_outputs)
        outputs = self.ffn(outputs)
        return outputs

class Decoder(nn.Module):

    def __init__(self, vocab_size, embed_dim, max_seq_len, n_heads, dropout_rate, n_layers):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddding = PositionalEmbedding(max_seq_len, embed_dim)
        self.decoder_layers = nn.ModuleList([DecoderLayer(embed_dim, n_heads, dropout_rate) for _ in range(n_layers)])

    def forward(self, enc_outputs, dec_x):
        """

        :param x:  shape=(batch_size, max_seq_len)
        :return:
        """

        # (batch_size, max_seq_len, embed_dim)
        dec_inputs = self.embedding(dec_x)

        # (batch_size, max_seq_len, embed_dim)
        dec_inputs += self.position_embeddding(dec_inputs)

        for decoder_layer in self.decoder_layers:
            dec_inputs = decoder_layer(enc_outputs, dec_inputs)

        return dec_inputs

class Transformer(nn.Module):

    def __init__(self, vocab_size, embed_dim, max_seq_len, n_heads, out_dim, dropout_rate, n_layers):

        super(Transformer, self).__init__()
        self.encoder = Encoder(vocab_size, embed_dim, max_seq_len, n_heads, dropout_rate, n_layers)
        self.decoder = Decoder(vocab_size, embed_dim, max_seq_len, n_heads, dropout_rate, n_layers)
        self.fc = nn.Linear(embed_dim, out_dim)

    def forward(self, enc_x, dec_x):
        enc_outputs = self.encoder(enc_x)
        dec_outputs = self.decoder(enc_outputs, dec_x)
        logits = self.fc(dec_outputs)
        return logits

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import math
#
#
# max_seq_len = 100
# embed_dim = 10
#
#
# positional_encoding = get_positional_encoding(100, 10)
#
# x , y = get_positional_encoding(100, 10)
#
# plt.figure(figsize=(10, 10))
# sns.heatmap(positional_encoding)
# plt.show()
#
# plt.figure(figsize=(10, 10))
# sns.heatmap(embedding_table)
# plt.show()
#
# embedding_table[0]
# positional_encoding[1]
