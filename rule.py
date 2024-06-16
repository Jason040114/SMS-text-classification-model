import pandas as pd
from sklearn.metrics import f1_score
# 加载数据集，注意sep参数表⽰更换分隔符
dataset = pd.read_csv("sms.csv")

# 前5000句作为训练集，5001到末尾作为验证集
train_dataset = dataset.head(5000)
valid_dataset = dataset.tail(-5000)
def rule():
    spam=['sex','urgent', 'important', 'claim', 'won', 'win', 'free', 'guaranteed', 'congratulations', 'prize', 'cash', 'reward', 'offer', 'limited time', 'act now', 'exclusive', 'opportunity', 'investment', 'stock', 'penny', 'profit', 'get rich', 'earn', 'income', 'extra income', 'home-based', 'work from home', 'no experience', 'financial freedom', 'credit score', 'consolidate', 'debt', 'lower your', 'mortgage', 'refinance', 'bankruptcy', 'cheap', 'discount', 'lowest price', 'no cost', 'no fees', 'pre-approved', 'risk-free', 'satisfaction guaranteed', 'privacy', 'removal', 'stop', 'unsubscribe', 'click here', 'link', 'confirm', 'password', 'account', 'bank', 'credit card', 'loan', 'paypal', 'social security', 'IRS', 'tax', 'lawsuit', 'legal action', 'phishing', 'spoof', 'scam']
    h=0
    predicted_labels=[]
    for i in range(5000,5000+len(valid_dataset)):
        s=0
        for j in spam:
            if j in valid_dataset.loc[i,'sms'].lower():
                s=1
                break
        predicted_labels.append(s)
        if s!=valid_dataset.loc[i,'label']:
            h=h+1
    score=f1_score(valid_dataset.label,predicted_labels)
    print(score)
