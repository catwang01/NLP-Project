import pytest
import torch
import unittest
from transformer import ScaledDotProductAttention, MultiHeadAttention, PositionalEmbedding, FFN, EncoderLayer, Encoder, padding_mask, DecoderLayer, Decoder, Transformer

class TestTransformer(unittest.TestCase):

    def test_scaled_dot_product_attention(self):
        torch.manual_seed(123)
        batch_size, max_seq_len, embed_dim = 2, 10, 512
        q = torch.randn(batch_size, max_seq_len, embed_dim)
        k = torch.randn(batch_size, max_seq_len, embed_dim)
        v = torch.randn(batch_size, max_seq_len, embed_dim)
        dot_product = ScaledDotProductAttention(0.5)
        assert dot_product(q, k, v).shape == (batch_size, max_seq_len, embed_dim)

    def test_multi_head_attention(self):

        batch_size, max_seq_len, embed_dim = 2, 10, 512
        embed_dim = 512
        n_heads = 8
        multihead = MultiHeadAttention(embed_dim, n_heads)

        torch.manual_seed(123)
        q = torch.randn(batch_size, max_seq_len, embed_dim)
        k = torch.randn(batch_size, max_seq_len, embed_dim)
        v = torch.randn(batch_size, max_seq_len, embed_dim)

        x = multihead(q, k, v)
        assert x.shape == (batch_size, max_seq_len, embed_dim)

    def test_positional_embedding(self):
        torch.manual_seed(123)
        batch_size, max_seq_len, embed_dim = 2, 10, 512
        positional_embedding = PositionalEmbedding(max_seq_len, embed_dim)
        x = torch.randn(batch_size, max_seq_len, embed_dim)
        assert positional_embedding(x).shape == x.shape

    def test_ffn(self):
        d_in, d_mid = 512, 512
        batch_size, max_seq_len, embed_dim = 2, 10, 512
        x = torch.randn(batch_size, max_seq_len, embed_dim)
        ffn = FFN(d_in, d_mid)
        assert ffn(x).shape == x.shape

    def test_encoder_layer(self):
        embed_dim, n_heads, dropout_rate = 512, 8, 0.5
        encoder_layer = EncoderLayer(embed_dim, embed_dim // 2, n_heads, dropout_rate)

        batch_size, max_seq_len = 5, 10
        x = torch.randn(batch_size, max_seq_len, embed_dim)
        assert encoder_layer(x).shape == x.shape

    def test_encoder(self):
        max_seq_len, vocab_size, embed_dim, n_heads, dropout_rate, n_layers = 10, 200, 512, 8, 0.5, 6
        encoder =  Encoder(vocab_size, embed_dim, max_seq_len, n_heads, dropout_rate, n_layers)

        batch_size = 5
        x = torch.randint(0, vocab_size, size=(batch_size, max_seq_len))
        assert encoder(x).shape == (batch_size, max_seq_len, embed_dim)

    def test_padding_mask(self):
        x = torch.tensor([[1, 2, 0, 0],
                          [2, 4, 3, 0]], dtype=torch.int)
        expected = torch.tensor([[[0, 0, 1, 1],
                                  [0, 0, 1, 1],
                                  [0, 0, 1, 1],
                                  [0, 0, 1, 1]],
                                [[0, 0, 0, 1],
                                 [0, 0, 0, 1],
                                 [0, 0, 0, 1],
                                 [0, 0, 0, 1]]], dtype=torch.bool)
        assert torch.all(padding_mask(x) == expected)

    def test_decoder_layer(self):
        embed_dim, n_heads, dropout_rate = 512, 8, 0.5
        batch_size, seq_len = 5, 10
        decoder_layer = DecoderLayer(embed_dim, n_heads, dropout_rate=0.0)
        enc_inputs = torch.randn(batch_size, seq_len, embed_dim)
        dec_inputs = torch.randn(batch_size, seq_len, embed_dim)
        assert decoder_layer(enc_inputs, dec_inputs).shape == enc_inputs.shape

    def test_decoder(self):
        vocab_size, embed_dim, max_seq_len, n_heads, dropout_rate, n_layers = 100, 512, 10, 8, 0.1, 6
        decoder = Decoder(vocab_size, embed_dim, max_seq_len, n_heads, dropout_rate, n_layers)
        batch_size = 10
        enc_outputs = torch.randn(batch_size, max_seq_len, embed_dim)
        dec_x  = torch.randint(0, max_seq_len, size=(batch_size, max_seq_len))
        assert decoder(enc_outputs, dec_x).shape == enc_outputs.shape

    def test_transformer(self):
        vocab_size, embed_dim, max_seq_len, n_heads, dropout_rate, n_layers, out_dim = 100, 512, 10, 8, 0.1, 6, 2
        batch_size = 10

        transformer = Transformer(vocab_size, embed_dim, max_seq_len, n_heads, out_dim, dropout_rate, n_layers)
        enc_x = torch.randint(0, max_seq_len, size=(batch_size, max_seq_len))
        dec_x  = torch.randint(0, max_seq_len, size=(batch_size, max_seq_len))
        assert transformer(enc_x, dec_x).shape == (batch_size, max_seq_len, out_dim)



if __name__ == "__main__":
    pytest.main()
