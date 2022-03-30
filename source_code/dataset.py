import torch
import torch.nn.functional as F
import json
from transformers import BertTokenizer

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Dataset 상속
class Dataset_custom(Dataset): 
    def __init__(self, data):
        self.x = load_data(data)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx]

def load_data(data):
    with open(data, 'r') as f:
        return json.load(f)

def custom_collate_fn(tokenizer: BertTokenizer, batch):

    input_sentences = [sample[1] for sample in batch]
    input_labels = [sample[2] for sample in batch]

    batch_inputs = tokenizer(input_sentences, padding = True, return_tensors = "pt")
    batch_labels = make_batch(input_labels)

    return batch_inputs, batch_labels

def make_batch(labels):
    batch_labels = []
    # special_token = 0
    
    for label in labels:
        sample_label = [label]
        # sample_label = [special_token] + [label] + [special_token]
        # sample_label += [special_token] * max(0, max_len - len(sample_label))
        batch_labels.append(sample_label)
        
    return torch.tensor(batch_labels)
