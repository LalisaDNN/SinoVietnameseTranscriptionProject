import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import os
import yaml
import json 
import time

def main():
    st.set_page_config(page_title='Sino-Vietnamese text transcription')

    config_path = "Config_1"
    config = load_config(config_path)
    base_vocab, sino_viet_words = load_vocab()
    model = load_model(config, base_vocab, sino_viet_words)
    max_len = config['data_config']['max_len']

    st.title('Sino-Vietnamese text transcription')
    st.divider()

    sino_viet_sent = st.text_area('Enter Sino-Vietnamese text:', '', height=100)

    ground_truth = st.text_area('Your transcription:', '', height=100)
    if st.button("Submit"):
        if sino_viet_sent:
            sino_viet_input = [c for c in sino_viet_sent]
            unk_words = in_vocab(sino_viet_words, sino_viet_input)

            if len(unk_words) > 0:
                unk_words = ", ".join(unk_words)
                st.write("Cannot make transcription since these words are not recognized:\n")
                st.write(f":red-background[{unk_words}].")
            else: 
                start_time = time.time()
                pred_sent = infer(model, sino_viet_sent, max_len)
                end_time = time.time()
                duration = end_time - start_time

                if ground_truth:
                    temp_ground_truth = ground_truth.lower()
                    temp_ground_truth = temp_ground_truth.split(" ")
                    temp_pred_sent = pred_sent.split(" ")

                    if len(temp_ground_truth) != len(temp_pred_sent):
                        st.text("The length of your transcription does not seem correct.")
                        st.subheader("Vietnamese transcription:")
                        st.write(pred_sent)
                    else: 
                        highlighted_text = ''
                        for i in range(len(temp_pred_sent)):
                            mask = temp_pred_sent[i] != temp_ground_truth[i]
                            if mask:
                                highlighted_text += f'_:red-background[{temp_pred_sent[i]}]_ '
                            else:
                                highlighted_text += f'{temp_pred_sent[i]} '

                        st.subheader("Vietnamese transcription:")
                        st.write(highlighted_text)
                        st.text('Differences are highlighted red.')
                else:
                    st.subheader("Vietnamese transcription:")
                    st.write(pred_sent)

                st.text(f"Inference time: {duration:.3} seconds.")
        else:
            st.write("Please enter Sino-Vietnamese text!")
        

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

def in_vocab(sino_viet_words, sino_viet_input):
    unk_words = []
    for word in sino_viet_input:
        if word not in sino_viet_words:
            unk_words.append(word)
    return unk_words

# @st.cache_data 
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

# @st.cache_data
def split_sentence(sent, max_len=512):
    return [sent[i:i + max_len] for i in range(0, len(sent), max_len)]

# @st.cache_data
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

# @st.cache_data
def infer (model, sent, max_len):
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

@st.cache_data
def load_vocab():
    with open('vocab/vocab.json', 'r', encoding="utf8") as vocab_file, open('vocab/sino_viet_words.json', 'r', encoding="utf8") as words_file:
        base_vocab = json.load(vocab_file)
        sino_viet_words = json.load(words_file)

    return base_vocab, sino_viet_words

@st.cache_data
def load_config(config_path):
    config_file = os.path.join(config_path, "config.yml")
    with open(config_file,'r') as conf:
        config = yaml.load(conf, Loader=yaml.SafeLoader)

    return config

@st.cache_resource
def load_model(config, base_vocab, sino_viet_words):
    bertmodel = config['model_config']['bert_model'] 

    base_tokenizer = BertTokenizer.from_pretrained(bertmodel)
    base_tokenizer.add_tokens(sino_viet_words)

    base_model = BertModel.from_pretrained(bertmodel)
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

    model_load_path = None if config['training_config']['model_load_path'] == 'None' else config['training_config']['model_load_path']
    model.load_state_dict(torch.load(model_load_path))

    return model
 

if __name__ == "__main__":
    main()
