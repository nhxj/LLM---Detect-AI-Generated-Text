
# 导入 pandas 库，用于数据分析和处理
import pandas as pd

# 导入 json 库，用于处理 JSON 数据格式
import json

# 导入 sys 库，用于访问系统特定的参数和函数
import sys

# 导入 gc 库，用于控制垃圾回收器
import gc

# 导入 StratifiedKFold 类，用于执行分层 K 折交叉验证
from sklearn.model_selection import StratifiedKFold

# 导入 numpy 库，用于科学计算和线性代数
import numpy as np

# 导入 roc_auc_score 函数，用于计算特征曲线下的面积
from sklearn.metrics import roc_auc_score

# 导入 LGBMClassifier 类，用于训练和使用 LightGBM 模型
from lightgbm import LGBMClassifier

# 导入 TfidfVectorizer 类，用于将文本转换为 TF-IDF 特征
from sklearn.feature_extraction.text import TfidfVectorizer

# 从 tokenizers 库导入各种类和函数，用于创建和使用自定义分词器
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

# 导入 Dataset 类，用于以标准化的方式处理数据集
from datasets import Dataset

# 导入 tqdm 库，用于显示进度条
from tqdm.auto import tqdm

# 导入 PreTrainedTokenizerFast 类，用于使用 transformers 库中的快速分词器
from transformers import PreTrainedTokenizerFast

# 导入 SGDClassifier 类，用于训练和使用随机梯度下降模型
from sklearn.linear_model import SGDClassifier

# 导入 MultinomialNB 类，用于训练和使用多项式朴素贝叶斯模型
from sklearn.naive_bayes import MultinomialNB

# 导入 VotingClassifier 类，用于将多个分类器组合成一个
from sklearn.ensemble import VotingClassifier

'''
数据集介绍：
test_essays.csv:包含一组3篇需要分类为人工生成或机器生成的文章。
sample_submission.csv:演示了预期的提交文件格式，包含两列：id和label。
train_v2_drcat_02.csv:包含一个由44868篇文章组成的数据集，附带相应的标签
        （0表示人工生成，1表示机器生成）。该文件用作训练集，用于开发用于检测AI生成文本的机器学习模型。
'''
test = pd.read_csv('test_essays.csv')

train = pd.read_csv("train_v2_drcat_02.csv", sep=',')

sub = pd.read_csv('sample_submission.csv')

# 去重
train = train.drop_duplicates(subset=['text'])
train.reset_index(drop=True, inplace=True)

test = test.drop_duplicates(subset=['text'])
test.reset_index(drop=True, inplace=True)

# 将LOWERCASE标志设置为False。这意味着在分词之前，文本不会被转换为小写
LOWERCASE = False

# 下面的代码将VOCAB_SIZE设置为14000000。这意味着词汇表中的最大单词数将为1400万。
VOCAB_SIZE = 14000000

# 使用字节对编码分词器进行文本分词

# 使用字节对编码（Byte Pair Encoding，BPE）算法创建分词器对象
raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

# 通过应用Unicode标准化形式C（NFC）对文本进行标准化，并可选择将其转换为小写
raw_tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else [])

# 通过将文本拆分成字节来进行预分词
raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

# 定义用于下游任务的特殊标记
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]

# 创建一个训练器对象，该对象将在给定的词汇表大小和特殊标记上训练分词器
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)

# 从pandas数据框中加载测试数据集，并仅选择文本列
dataset = Dataset.from_pandas(test[['text']])


# 定义一个生成器函数，用于从数据集中生成文本批次
def train_corp_iter():
    for i in range(0, len(dataset), 1000):
        yield dataset[i: i + 1000]["text"]


# 使用训练器对象对文本批次进行分词器训练
raw_tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer)

# 将原始分词器对象封装为与HuggingFace库兼容的PreTrainedTokenizerFast对象
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw_tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

# 初始化一个空列表，用于存储测试集的分词后的文本
tokenized_texts_test = []

# 遍历测试集中的文本，并使用分词器对象对其进行分词
for text in tqdm(test['text'].tolist()):
    tokenized_texts_test.append(tokenizer.tokenize(text))

# 初始化一个空列表，用于存储训练集的分词后的文本
tokenized_texts_train = []

# 遍历训练集中的文本，并使用分词器对象对其进行分词
for text in tqdm(train['text'].tolist()):
    tokenized_texts_train.append(tokenizer.tokenize(text))


# 定义一个返回输入文本的虚拟函数
def dummy(text):
    return text


# 创建一个TfidfVectorizer对象，从文本中提取词的n-gram（3到5个词），不转换为小写或进行分词
vectorizer = TfidfVectorizer(ngram_range=(3, 5),
                             lowercase=False,
                             sublinear_tf=True,
                             analyzer='word',
                             tokenizer=dummy,
                             preprocessor=dummy,
                             token_pattern=None, strip_accents='unicode')

# 在测试集的分词文本上拟合向量化器
vectorizer.fit(tokenized_texts_test)

# 获取向量化器的词汇表，它是一个n-gram和它们的索引的字典
vocab = vectorizer.vocabulary_

# 创建另一个TfidfVectorizer对象，使用之前向量化器获得的词汇表，但参数保持不变
vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, vocabulary=vocab,
                             analyzer='word',
                             tokenizer=dummy,
                             preprocessor=dummy,
                             token_pattern=None, strip_accents='unicode')

# 在训练集的分词文本上拟合和转换向量化器，并获取tf-idf值的稀疏矩阵
tf_train = vectorizer.fit_transform(tokenized_texts_train)

# 在测试集的分词文本上转换向量化器，并获取tf-idf值的稀疏矩阵
tf_test = vectorizer.transform(tokenized_texts_test)

# 删除向量化器对象以释放内存
del vectorizer

# 调用垃圾回收器以回收未使用的内存
gc.collect()

y_train = train['label'].values


# 定义一个函数，返回一个由四个分类器组成的集成模型
def get_model():
    # 从catboost库中导入CatBoostClassifier
    from catboost import CatBoostClassifier

    # 创建一个Multinomial Naive Bayes分类器，平滑参数为0.0235
    clf = MultinomialNB(alpha=0.0225)

    # 创建一个Stochastic Gradient Descent分类器，最大迭代次数为9000，容差为3e-4，使用修改后的Huber损失函数，随机种子为6743
    sgd_model = SGDClassifier(max_iter=9000, tol=1e-4, loss="modified_huber", random_state=42)

    # 定义一个LightGBM分类器的参数字典
    p = {
        'verbose': -1,
        'n_iter': 3000,
        'colsample_bytree': 0.7800,
        'colsample_bynode': 0.8000,
        'random_state': 6743,
        'metric': 'auc',
        'objective': 'cross_entropy',
        'learning_rate': 0.00581909898961407,
    }

    # 使用给定的参数创建一个LightGBM分类器
    lgb = LGBMClassifier(**p)

    # 创建一个CatBoost分类器，迭代次数为6000，学习率为0.003599066836106983，子样本比例为0.4，使用交叉熵损失函数，随机种子为6543
    cat = CatBoostClassifier(iterations=3000,
                             verbose=0,
                             random_seed=42,
                             learning_rate=0.005599066836106983,
                             subsample=0.35,
                             allow_const_label=True,
                             loss_function='CrossEntropy'
                             )

    # 定义一个四个分类器的权重列表
    weights = [0.1, 0.31, 0.28, 0.67]

    # 创建一个投票分类器，使用软投票和并行处理，将四个分类器组合起来
    ensemble = VotingClassifier(estimators=[('mnb', clf),
                                            ('sgd', sgd_model),
                                            ('lgb', lgb),
                                            ('cat', cat)
                                            ],
                                weights=weights, voting='soft', n_jobs=-1)

    # 返回集成模型
    return ensemble


# 调用get_model函数，并将返回的模型赋值给一个变量
model = get_model()
model.fit(tokenized_texts_train, train['label'])

# 打印模型
print(model)

# 检查测试文本值的长度
if len(test.text.values) <= 5:
    # 如果长度小于等于5，将提交的数据框保存为csv文件
    sub.to_csv('submission.csv', index=False)
else:
    # 否则，将模型拟合于训练集的tf-idf矩阵和目标标签上
    model.fit(tf_train, y_train)

    # 调用垃圾回收器以回收未使用的内存
    gc.collect()

    # 使用模型预测测试集上正类别的概率
    final_preds = model.predict_proba(tf_test)[:, 1]

    # 将预测的概率赋值给提交数据框的生成列
    sub['generated'] = final_preds

    # 将提交数据框保存为csv文件
    sub.to_csv('submission.csv', index=False, encoding='utf-8')

    # 显示提交数据框
    sub.head()

from sklearn.metrics import roc_auc_score

# 假设 preds 是模型在测试集上的预测概率结果
preds = model.predict_proba(tokenized_texts_test)[:, 1]

roc_auc = roc_auc_score(test['label'], preds)
print(f"ROC AUC: {roc_auc}")
