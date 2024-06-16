import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import torch.optim as optim
import re
import nltk
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("sms.csv")
train_df = dataset[:5000]
test_df = dataset[5000:]
def tokenizer(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    t = text.split()
    t = [i for i in t if i.isalpha()]
    t = [WordNetLemmatizer().lemmatize(i) for i in t]
    return t
def CNN_PCA():
    vec = TfidfVectorizer(stop_words="english", min_df=0, max_df=1.0, tokenizer=tokenizer)
    X_train = vec.fit_transform(train_df['sms']).toarray()
    feature=len(X_train[0])
    y_train = train_df['label'].values
    X_test = vec.transform(test_df['sms']).toarray()
    y_test = test_df['label'].values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 创建PCA对象并指定降维后的维度
    pca = PCA(n_components=1000)

    # 在训练集上拟合PCA模型并进行降维
    X_train = pca.fit_transform(X_train_scaled)

    # 在测试集上进行降维
    X_test = pca.transform(X_test_scaled)

    # 更新特征维度
    feature = X_train.shape[1]
    class SMSDataset(Dataset):
        def __init__(self, data, labels):
            self.data = torch.tensor(data, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3,padding=1)
            self.pool1 = nn.MaxPool1d(kernel_size=2)
            self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,padding=1)
            self.pool2 = nn.MaxPool1d(kernel_size=2)
            self.fc1 = nn.Linear(64 * int(feature/4), 512)
            self.fc2 = nn.Linear(512, 2)

        def forward(self, x):
            x = x.unsqueeze(1)
            x = self.conv1(x)
            x = nn.functional.relu(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = nn.functional.relu(x)
            x = self.pool2(x)
            x = x.view(-1, 64 *int(feature/4))
            x = self.fc1(x)
            x = nn.functional.relu(x)
            x = self.fc2(x)
            return x

    # 定义超参数和优化器
    batch_size = 128
    num_epochs = 39
    learning_rate = 0.001
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    train_set = SMSDataset(X_train, y_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_set = SMSDataset(X_test, y_test)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)


    for epoch in range(num_epochs):
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                y_pred.extend(predicted.tolist())
                y_true.extend(labels.tolist())
        f1 = f1_score(y_true, y_pred)
        print(f1)