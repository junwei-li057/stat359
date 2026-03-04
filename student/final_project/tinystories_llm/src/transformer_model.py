import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyStoriesConfig:
    def __init__(
        self,
        vocab_size=10000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=1024,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        window_size=256,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.window_size = window_size
        self.model_type = "causal_lm"
        self.tie_word_embeddings = True

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)

    def to_dict(self):
        return self.__dict__

class TinyStoriesEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        
    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = self.position_ids[:, :seq_length]
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class TinyStoriesSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(context_layer.size()[:-2] + (self.all_head_size,))
        
        attention_output = self.output(context_layer)
        return attention_output, attention_probs

class TinyStoriesFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states):
        hidden_states = self.dense1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.dense2(hidden_states)
        return self.dropout(hidden_states)

class TinyStoriesLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = TinyStoriesSelfAttention(config)
        self.feed_forward = TinyStoriesFeedForward(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        attn_out, attn_probs = self.attention(hidden_states, attention_mask)
        hidden_states = residual + attn_out
        
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        ffn_out = self.feed_forward(hidden_states)
        hidden_states = residual + ffn_out
        return hidden_states, attn_probs

class TinyStoriesModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = TinyStoriesEmbeddings(config)
        self.layers = nn.ModuleList([TinyStoriesLayer(config) for _ in range(config.num_hidden_layers)])
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def _get_causal_mask(self, seq_length, device):
        mask = torch.full((seq_length, seq_length), float('-inf'), device=device)
        for i in range(seq_length):
            start = max(0, i - self.config.window_size + 1)
            mask[i, start:i+1] = 0.0
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids=None, attention_mask=None, output_attentions=False):
        hidden_states = self.embeddings(input_ids)
        causal_mask = self._get_causal_mask(input_ids.size(1), input_ids.device)
        
        if attention_mask is not None:
            mask_ = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0
            combined_mask = causal_mask + mask_
        else:
            combined_mask = causal_mask

        all_attentions = []
        for layer in self.layers:
            hidden_states, attn_probs = layer(hidden_states, combined_mask)
            if output_attentions: all_attentions.append(attn_probs)
        
        hidden_states = self.layer_norm(hidden_states)
        return {"last_hidden_state": hidden_states, "attentions": all_attentions}

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

class TinyStoriesForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = TinyStoriesModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.get_input_embeddings().weight

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, **kwargs}

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.transformer(input_ids, attention_mask)
        logits = self.lm_head(outputs["last_hidden_state"])
        return {"logits": logits}

    def generate(self, input_ids, max_length, temperature=1.0, top_p=0.9, **kwargs):
        for _ in range(max_length - input_ids.size(1)):
            logits = self.forward(input_ids)["logits"][:, -1, :]
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        return input_ids