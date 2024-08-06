import torchtext
torchtext.disable_torchtext_deprecation_warning()

import time
import torch
import torch.nn as nn
from torchtext.datasets import AG_NEWS
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from .prepare_funcs import (neural_networks, Prediction, HardPredict)

class train_text:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_train_iter = AG_NEWS(split="train")
    num_class = len(set([label for (label, text) in text_train_iter]))
    emsize = 64
    
    epochs = 10
    LR = 5
    BATCH_SIZE = 64
    
    criterion = nn.CrossEntropyLoss()
    total_accu = None
    train_iter, test_iter = AG_NEWS()
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)
    num_train = int(len(train_dataset) * 0.95)
    split_train, split_valid = random_split(
        train_dataset, [num_train, len(train_dataset) - num_train]
    )
    
    def __init__(self, tokenizer: str = "basic_english") -> None:
        self.tokenizer = get_tokenizer(tokenizer)
    
    def __yield_tokens(self, data_iter):
        for _, text in data_iter:
            yield self.tokenizer(text)
    
    
    def __prepare_vocab(self, train_iter, specials_):
        yield_tokens = self.__yield_tokens(train_iter)
        vocab = build_vocab_from_iterator(yield_tokens, specials=specials_)
        vocab.set_default_index(vocab["<unk>"])
        return vocab
        
    def text_pipeline(self, text):
        vocab = self.__prepare_vocab(self.text_train_iter, ["<unk>"])
        return vocab(self.tokenizer(text))
    
    label_pipeline = lambda x: int(x) - 1
    
    def collate_batch(self, batch):
        label_list, text_list, offsets = [], [], [0,]
        for _label, _text in batch:
            label_list.append(self.label_pipeline(_text))
            processed_text = torch.tensor(self.text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(self.device), text_list.to(self.device), offsets.to(self.device)
    
    class TextClassificationModel(nn.Module):
        def __init__(self, vocab_size, embed_dim, num_class):
            super(train_text.TextClassificationModel, self).__init__()
            self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
            self.fc = nn.Linear(embed_dim, num_class)
            self.init_weights()
        
        def init_weights(self):
            initrange = 0.5
            self.embedding.weight.data.uniform_(-initrange, initrange)
            self.fc.weight.data.uniform_(-initrange, initrange)
            self.fc.bias.data.zero_()
        
        def forward(self, text, offsets):
            embedded = self.embedding(text, offsets)
            return self.fc(embedded)
        
    # TODO: Try to fix these dataloaders accesibility
    # PROBLEM: `collate_batch` 
    
    def prepare_dataloaders(self):
        
        train_dataloader = DataLoader(
            self.split_train, batch_size=self.BATCH_SIZE, shuffle=True, collate_fn=self.collate_batch
        )
        test_dataloader = DataLoader(
            self.split_train, batch_size=self.BATCH_SIZE, shuffle=True, collate_fn=self.collate_batch
        )
        valid_dataloader = DataLoader(
            self.split_train, batch_size=self.BATCH_SIZE, shuffle=True, collate_fn=self.collate_batch
        )
        
        dictionary = dict(train=train_dataloader, test=test_dataloader, validation=valid_dataloader)
        return dictionary
    
    def prepare_items(self):
        vocab_size = len(self.__prepare_vocab(self.text_train_iter, ["<unk>"]))
        model = self.TextClassificationModel( vocab_size, self.emsize, self.num_class ).to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.LR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
        dictionary = dict(vocab_size=vocab_size, model=model, 
                          optimizer=optimizer, scheduler=scheduler)
        return dictionary
    
    def train(self, dataloader):
        items = self.prepare_items()
        model, optimizer = items.get("model"), items.get("optimizer")
        model.train()
        
        total_acc, total_count = 0, 0
        long_interval = 500
        start_time = time.time()
        
        for idx, (label, text, offsets) in enumerate(dataloader):
            optimizer.zero.grad()
            predicted_label = model(text, offsets)
            loss = self.criterion(predicted_label, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            if idx % long_interval == 0 and idx > 0:
                print(
                    "| epoch {:3d} | {:5d}/{:5d} batches ",
                    "| accuracy {:8.3f}".format(
                        self.epoch, idx, len(dataloader), total_acc / total_count
                    ),
                    sep="\n"
                )
                total_acc, total_count = 0, 0
                start_time = time.time()
    
    def evaluate(self, dataloader):
        items = self.prepare_items()
        model = items["model"]
        model.eval()
        total_acc, total_count = 0, 0
        
        with torch.no_grad():
            for idx, (label, text, offsets) in enumerate(dataloader):
                predicted_label = model(text, offsets)
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
            return total_acc / total_count
    
    def predict(self, text):
        items = self.prepare_items()
        model = items["model"]
        text_pipeline = self.text_pipeline(text)
        with torch.no_grad():
            text = torch.tensor(text_pipeline)
            output = model(text, torch.tensor([0,]))
            return output.argmax(1).item() + 1
    
    def run_model(self):
        items = self.prepare_items()
        dataloaders = self.prepare_dataloaders()
        train_dataloader = dataloaders["train"]
        valid_dataloader = dataloaders["validation"]
        scheduler = items["optimizer"], items["scheduler"]
        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.time()
            self.train(train_dataloader)
            accu_val = self.evaluate(valid_dataloader)
            if self.total_accu != None and self.total_accu > accu_val:
                scheduler.step()
            else:
                self.total_accu = accu_val
            
            print("-" * 59)
            print(
                "| end of epoch {:3d} | time: {:5.2f}s |",
                "valid accuracy {:8.3f} ".format(
                    epoch, time.time() - epoch_start_time, accu_val
                ),
                sep="\n"
            )
            print("-" * 59)


class Train_Image:
    def __init__(self) -> None: ...


class mathematics:
    def __init__(self, in_categorie, out_categorie) -> None:
        self.in_categorie = in_categorie
        self.out_categorie = out_categorie
    
    def __init_instance(self, input_data, input_units, input_shape):
        in_categorie, out_categorie = self.in_categorie, self.out_categorie
        neural_net = neural_networks( in_categorie, out_categorie )
        tf_neural_net = neural_net.tensor_flow(
            rounds=400, optimizer_value=1.0,
            input_data=input_data, input_shape=input_shape,
            input_units=input_units
            )
        return tf_neural_net
    
    def precission(self, value, expected_value, input_units, input_shape):
        
        instance = self.__init_instance(value, input_units, input_shape)
        
        x, y = instance["result"], expected_value
        result = ( abs(y - x) / y ) * 10
        adjust_array = result[0] # adjust from [[x]] to [x]
        
        message = ...
        
        if adjust_array[0] >= 5:
            message = "The precission is so good!"
            
        elif adjust_array[0] > 8:
            message = "The precission is so excelent!"
        
        else:
            message = "So bad precission"
            
        return adjust_array[0], message