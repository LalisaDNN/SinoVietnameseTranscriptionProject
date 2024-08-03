import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import os
import yaml
import json 
import time

####### Config #######
config_path = "Config_1"
config_file = os.path.join(config_path, "config.yml")
with open(config_file,'r') as conf:
    config = yaml.load(conf, Loader=yaml.SafeLoader)

class AddNorm(nn.Module):
    def __init__(self, norm_shape: int, dropout=0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_ff_dim: int, dropout=0.3):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_ff_dim, input_dim)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    
class ShrinkNorm(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout=0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim, output_dim)
        self.ln = nn.LayerNorm(output_dim)

    def forward(self, x):
        return self.ln(self.dropout(self.linear(x)))
    
class SinoVietnameseTranslator(nn.Module):
    def __init__(self, tokenizer, base_model, vocab, hidden_ff_dim=512, model_hidden_dim=512, 
                 large_hidden_classification_head_dim=256, small_hidden_classification_head_dim=128,
                 max_num_spellings=7, num_spelling_threshold=3, train_bert_param=True):
        super(SinoVietnameseTranslator, self).__init__()
        self.tokenizer = tokenizer
        self.bert = base_model
        self.vocab = vocab
        self.max_num_spellings = max_num_spellings
        
        for param in self.bert.parameters():
            param.requires_grad = train_bert_param
        
        self.shrink_norm = ShrinkNorm(self.bert.config.hidden_size, model_hidden_dim)
        self.feed_forward = FeedForwardNetwork(model_hidden_dim, hidden_ff_dim)
        self.add_norm = AddNorm(model_hidden_dim)
        
        self.classification_heads = nn.ModuleDict()
        for sino_word, viet_spellings in self.vocab.items():
            if len(viet_spellings) > 1 and len(viet_spellings) <= num_spelling_threshold:
                num_spellings = len(viet_spellings)
                self.classification_heads[sino_word] = nn.Sequential(
                    nn.Linear(model_hidden_dim, small_hidden_classification_head_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(small_hidden_classification_head_dim, num_spellings),
                    nn.Softmax(dim=-1)
                )
            elif len(viet_spellings) > num_spelling_threshold:
                num_spellings = len(viet_spellings)
                self.classification_heads[sino_word] = nn.Sequential(
                    nn.Linear(model_hidden_dim, large_hidden_classification_head_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(large_hidden_classification_head_dim, num_spellings),
                    nn.Softmax(dim=-1)
                )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        shrink_output = self.shrink_norm(sequence_output)
        projected_output = self.add_norm(shrink_output, self.feed_forward(shrink_output))
        
        batch_size, max_len = input_ids.size()
        predictions = torch.full((batch_size, max_len, self.max_num_spellings), -1.0, device=input_ids.device)
        
        for i in range(batch_size):
            for j in range(max_len):
                token_id = input_ids[i, j].item()
                if token_id == self.tokenizer.pad_token_id:
                    continue
                    
                sino_word = self.tokenizer.convert_ids_to_tokens(token_id)
                
                if sino_word in self.classification_heads:
                    logits = self.classification_heads[sino_word](projected_output[i, j])
                    predictions[i, j, :len(logits)] = logits
                else:
                    predictions[i, j, 0] = 1.0

        return predictions
    
def decode_predictions(predictions, input_ids, tokenizer, vocab):
    decoded_sentences = []
    for i, predicted_indices in enumerate(predictions):
        decoded_sentence = []
        for j, spelling_index in enumerate(predicted_indices):
            token = input_ids[i, j].item()
            if token == tokenizer.pad_token_id:
                continue
                
            sino_word = tokenizer.convert_ids_to_tokens(token)
            viet_spelling = vocab[sino_word][spelling_index]
            decoded_sentence.append(viet_spelling)

        decoded_sentences.append(" ".join(decoded_sentence))
    return " ".join(decoded_sentences)

def split_sentence(sent, max_len=512):
    return [sent[i:i + max_len] for i in range(0, len(sent), max_len)]

def prepare_input(sent, tokenizer, max_len=512):
    chunks = split_sentence(sent, max_len)
    
    input_ids_chunks = []
    attention_mask_chunks = []

    for sentence in chunks:
        tokens = tokenizer.encode(sentence, add_special_tokens=False, max_length=max_len, truncation=True)
        input_ids = tokens + [tokenizer.pad_token_id] * (max_len - len(tokens))
        attention_mask = [1] * len(tokens) + [0] * (max_len - len(tokens))

        input_ids_chunks.append(input_ids)
        attention_mask_chunks.append(attention_mask)

    return torch.tensor(input_ids_chunks), torch.tensor(attention_mask_chunks)

def infer (sent, model, max_len):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)    

    tokenizer = model.tokenizer
    vocab = model.vocab

    model.eval()
    with torch.no_grad():
        input_ids, attention_mask = prepare_input(sent=sent, tokenizer=tokenizer, max_len=max_len)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = model(input_ids, attention_mask=attention_mask)

        predictions = torch.argmax(outputs, dim=-1)

        translated_sent = decode_predictions(predictions, input_ids, tokenizer, vocab)
    
    return translated_sent

with open('vocab/vocab.json', 'r', encoding="utf8") as vocab_file, open('vocab/sino_viet_words.json', 'r', encoding="utf8") as words_file:
    base_vocab = json.load(vocab_file)
    sino_viet_words = json.load(words_file)
    
# Model Config
bert_model = config['model_config']['bert_model'] 

base_tokenizer = BertTokenizer.from_pretrained(bert_model)
base_tokenizer.add_tokens(sino_viet_words)

base_model = BertModel.from_pretrained(bert_model)
base_model.resize_token_embeddings(len(base_tokenizer))

# Model config
hidden_ff_dim = config['model_config']['hidden_ff_dim']
model_hidden_dim = config['model_config']['model_hidden_dim']
large_hidden_classification_head_dim = config['model_config']['large_hidden_classification_head_dim']
small_hidden_classification_head_dim = config['model_config']['small_hidden_classification_head_dim']
max_num_spellings = config['model_config']['max_num_spellings']
num_spelling_threshold = config['model_config']['num_spelling_threshold']
train_bert_param = config['model_config']['train_bert_param']

model = SinoVietnameseTranslator(base_tokenizer, base_model, base_vocab, 
                                hidden_ff_dim=hidden_ff_dim, model_hidden_dim=model_hidden_dim,
                                large_hidden_classification_head_dim=large_hidden_classification_head_dim,
                                small_hidden_classification_head_dim=small_hidden_classification_head_dim,
                                max_num_spellings=max_num_spellings, train_bert_param=train_bert_param,
                                num_spelling_threshold=num_spelling_threshold)

print(sum([param.nelement() for param in model.parameters()]))

model_load_path = None if config['training_config']['model_load_path'] == 'None' else config['training_config']['model_load_path']
model.load_state_dict(torch.load(model_load_path))

max_len = config['data_config']['max_len']

sino_viet_sent = str(input("Input Sino-Vietnamese sentence: "))

start_time = time.time()
pred_sent = infer(sino_viet_sent, model, max_len)
end_time = time.time()
duration = end_time - start_time

print(f"Vietnamese spelling: {pred_sent}")
print(f"Inference time: {duration:.5} seconds.")

