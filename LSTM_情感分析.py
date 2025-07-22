# %%
import os
import torch
import jieba
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from sklearn import metrics
import json
import pandas as pd
import time
from datetime import timedelta


# %% [markdown]
# 

# %%
MAX_LEN =  100
TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 256
VOCAB_SIZE = len(json.load(open('./data/word2id.json', 'r', encoding='utf-8')))
learning_rate = 1e-4
NUM_EPOCH = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% [markdown]
# # 数据处理

# %%
class SentimentDataset(Dataset):
    def __init__(self, data_path: str, data_type: str, max_token_len: int = 512):
        self.max_token_len = max_token_len
        self.data = json.load(open(data_path))
        self.word2id =  json.load(open('./data/word2id.json', 'r', encoding='utf-8'))
        self.label2id = {'angry':0, 'neutral':1, 'happy':2, 'sad':3, 'surprise':4, 'fear':5}

    def transform(self, sentence):
        sent_ids = []
        sentence = jieba.lcut(sentence)
        for word in sentence:
            if not (word in self.word2id.keys()):
                sent_ids.append(1)
            else:
                sent_ids.append(self.word2id[word])
        return sent_ids
    
    def __getitem__(self, index: int):
        line = self.data[index]
        content = line['content']
        label = line['label']

        sentence_id = self.transform(content)
        label_id = self.label2id[label]
    
        return sentence_id,label_id

    def __len__(self):
        return len(self.data)

# %%

train_dataset = SentimentDataset("./data/usual_train.txt", 'train',  max_token_len = MAX_LEN)
eval_dataset = SentimentDataset("./data/usual_eval_labeled.txt", 'eval',  max_token_len= MAX_LEN)

# %%
for td  in train_dataset:
    print(td)
    break

# %%
def collate_fn(batch):
    sentence_ids  = [torch.LongTensor(sentence_id) for sentence_id,label_id in batch]
    label_ids  = [torch.LongTensor([label_id]) for sentence_id,label_id in batch]
    # 将文本序列填充至同一长度
    sentence_ids = pad_sequence(sentence_ids, batch_first=True, padding_value=0)
    label_ids = torch.LongTensor(label_ids)
    return sentence_ids,label_ids

# %%
train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0, collate_fn = collate_fn)
eval_dataloader = DataLoader(eval_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=0, collate_fn = collate_fn)


# %%
for td in train_dataloader:
    print(td)
    break

# %% [markdown]
# # 模型构建

# %%
# LSTMWithGlobalMaxPooling
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim = 300, hidden_dim = 128, n_layers = 2, bidirectional=False, dropout = 0.1, pad_idx = None, num_labels =6):
        super().__init__()

        # embedding层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # lstm层
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers = n_layers,
            bidirectional = bidirectional,
            dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

        # 输出层
        # 全连接层
        num_direction = 2 if bidirectional else 1   # 双向&单向可选
        self.fc = nn.Linear(hidden_dim  * num_direction, num_labels)
        
    def forward(self, text):

        embedded = self.embedding(text)   # [ batch size, sent len] -> [sent len, batch size, emb dim]


        hidden_output,(h_n, c_n) = self.lstm(embedded)

        pooled_ouput, _ = torch.max(hidden_output, dim=1)  
        
        loggits = self.fc(pooled_ouput)
        
        return loggits

# %%
model = LSTMModel(
    vocab_size= VOCAB_SIZE,
    embedding_dim = 300, 
    hidden_dim = 128,
    n_layers = 1,
    bidirectional=True,
    dropout = 0.1,
    pad_idx = 0,
    num_labels = 6)

# %% [markdown]
# 

# %%
model = model.to(device)

# %% [markdown]
# # 模型训练

# %% [markdown]
# 

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dev_best_loss = float('inf')
writer = SummaryWriter(log_dir='./lstm_v2/')
NUM_EPOCH= 20

for epoch in range(NUM_EPOCH):
    model.train()
    train_total_loss = 0.
    nb_train_steps = 0
    train_labels_list, train_preds_list = [], []
    for sentence_ids,label_ids in train_dataloader:                     # 遍历一个epoch的数据
        nb_train_steps +=1
        sentence_ids, label_ids = sentence_ids.to(device) ,label_ids.to(device)  # 加载到GCP上
        outputs = model(sentence_ids)                       # 前向传播
        train_loss = criterion(outputs, label_ids)      # 计算损失
        train_total_loss += train_loss.item()
        optimizer.zero_grad()                            # 梯度清零
        train_loss.backward()                            # 反向传播计算梯度
        optimizer.step()                                 # 更新梯度
        train_preds_list.extend(torch.max(outputs.data, 1)[1].cpu().tolist())
        train_labels_list.extend(label_ids.data.cpu().tolist())

  
    train_acc = accuracy_score(train_labels_list, train_preds_list)
    train_loss = train_total_loss/nb_train_steps
    print(f'epoch:{epoch}, train_loss:{train_loss}, train_acc:{train_acc}')


    model.eval()
    labels_list, preds_list = [], []
    nb_eval_steps = 0
    eval_total_loss = 0.
    with torch.no_grad():
        for sentence_ids,label_ids in eval_dataloader:
            sentence_ids,label_ids = sentence_ids.to(device) ,label_ids.to(device)
            outputs = model(sentence_ids)

            eval_loss = criterion(outputs, label_ids)      # 计算损失
            eval_total_loss += eval_loss.item()
            
            preds_list.extend(torch.max(outputs.data, 1)[1].cpu().tolist())
            labels_list.extend(label_ids.data.cpu().tolist())

            nb_eval_steps +=1

    dev_acc = accuracy_score(labels_list, preds_list)
    dev_mp, dev_mr, dev_mf, _ =precision_recall_fscore_support(
                    y_true=labels_list, y_pred=preds_list, average='macro')

    dev_loss = eval_total_loss / nb_eval_steps

    if dev_loss < dev_best_loss:
        dev_best_loss = dev_loss
        torch.save(model.state_dict(), "./cache/TextRNN_model.bin")

    writer.add_scalar("loss/train", train_loss, epoch)
    writer.add_scalar("loss/dev", dev_loss, epoch)
    writer.add_scalar("acc/train", train_acc, epoch)
    writer.add_scalar("acc/dev", dev_acc, epoch)
    writer.add_scalar("mp/dev", dev_mp, epoch)
    writer.add_scalar("mr/dev", dev_mr, epoch)
    writer.add_scalar("mf/dev", dev_mf, epoch)


    print(f'epoch:{epoch}, dev_loss:{dev_loss} dev_acc: {dev_acc}, dev_macro_precision:{dev_mp}, dev_macro_recall:{dev_mr}, dev_macro_f1:{dev_mf}')

# %%


# %% [markdown]
# # 模型预测

# %%
test_dataset = SentimentDataset("./data/usual_test_labeled.txt", 'test',  max_token_len= MAX_LEN)
test_dataloader = DataLoader(test_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=0, collate_fn = collate_fn)


# %%


# test
model.load_state_dict(torch.load("./cache/TextRNN_model.bin"))
model.eval()

test_total_loss = 0.
labels_list, preds_list = [], []
for sentence_ids,label in test_dataloader:
    sentence_ids = sentence_ids.to(device)
    label = label.to(device)
    outputs = model(sentence_ids)

    test_loss = criterion(outputs, label)      # 计算损失
    test_total_loss += test_loss.item()

    preds_list.extend(torch.max(outputs.data, 1)[1].cpu().tolist())
    labels_list.extend(label.data.cpu().tolist())

test_acc = metrics.accuracy_score(labels_list, preds_list)

test_report = metrics.classification_report(labels_list, preds_list, digits=6)
test_confusion = metrics.confusion_matrix(labels_list, preds_list)

msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
print(msg.format(test_loss, test_acc))
print("Precision, Recall and F1-Score...")
print(test_report)
print("Confusion Matrix...")
print(test_confusion)




# %%



