import torch
import torch.nn.functional as F
import torch.nn as nn


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention *= scale
        if attn_mask:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = F.softmax(attention, dim=2)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, d_model, n_heads, dropout_rate=0.0):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.dk = d_model // n_heads
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_q = nn.Linear(d_model, self.dk * n_heads)
        self.linear_k = nn.Linear(d_model, self.dk * n_heads)
        self.linear_v = nn.Linear(d_model, self.dk * n_heads)
        self.scaled_dot_product_attention = ScaledDotProductAttention(dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear_final = nn.Linear(self.dk * n_heads, self.dk * n_heads)
        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant(m.weight, 1.0)
                nn.init.constant(m.bias, 0)

    def forward(self, Q, K, V, attn_mask=None):
        scale = self.d_model ** (-0.5)
        q = self.linear_q(Q)
        k = self.linear_k(K)
        v = self.linear_v(V)
        z = self.scaled_dot_product_attention(q, k, v, scale, attn_mask)

        residual = self.linear_final(z)
        output = self.layer_norm(Q + residual)
        return output


multihead2 = MultiHeadAttention(512, 8)

torch.manual_seed(123)
q = torch.randn(2, 10, 512)
k = torch.randn(2, 10, 512)
v = torch.randn(2, 10, 512)

x2 = multihead2(q, k, v)
