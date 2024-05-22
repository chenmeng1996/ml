from re import S
from distro import like
from nltk.stem import WordNetLemmatizer
from nltk.corpus import names
import glob
import os
import numpy as np

# 两个数组，分别存储邮件内容，邮件标签（1表示垃圾邮件，0表示非垃圾邮件）
e_mails, labels = [], []

"""读数据"""
file_path = 'spam_identification/enron1/spam/'
for filename in glob.glob(os.path.join(file_path, '*.txt')):
    with open(filename, 'r', encoding="ISO-8859-1") as infile:
        e_mails.append(infile.read())
        labels.append(1)

file_path = 'spam_identification/enron1/pam/'
for filename in glob.glob(os.path.join(file_path, '*.txt')):
    with open(filename, 'r', encoding="ISO-8859-1") as infile:
        e_mails.append(infile.read())
        labels.append(1)

def letters_only(astr):
    return astr.isalpha()

all_names = set(names.words())
# 词形还原器
lemmatizer = WordNetLemmatizer()

def clean_text(docs):
    """
    数据清洗
    - 删除数字和标点
    - 删除人名（可选）
    - 词形还原
    """
    cleaned_docs = []
    for doc in docs:
        cleaned_docs.append(
            ' '.join([lemmatizer.lemmatize(word.lower())
                        for word in doc.split()
                        if letters_only(word)
                        and word not in all_names]))
    return cleaned_docs

cleaned_e_mails = clean_text(e_mails)


"""
特征提取

1. 去除停用词
2. 计算词频
3. 只考虑词频在500以上的单词作为特征
4. 转换成稀疏矩阵，其中每一行表示一个邮件，每一列表示一个单词的索引，值是该单词在该邮件中出现的次数
"""
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words="english", max_features=500)
# 词频特征矩阵，行是邮件，列是单词，值是单词在邮件中出现的次数
term_docs = cv.fit_transform(cleaned_e_mails)

def get_label_index(labels):
    from collections import defaultdict
    label_index = defaultdict(list)
    for index, label in enumerate(labels):
        label_index[label].append(index)
    return label_index

# 按照标签将邮件分成两类：垃圾邮件和非垃圾邮件以及对应的邮件索引数组
label_index = get_label_index(labels)

def get_prior(label_index):
    """
    基于训练数据，计算先验概率

    垃圾邮件事件为S, 非垃圾邮件事件为NS
    P(S) = 垃圾邮件数 / 总邮件数
    P(NS) = 非垃圾邮件数 / 总邮件数

    Returns:
        字典, key是标签, value是该标签的先验概率
    """
    prior = {label: len(index) for label, index in label_index.items()}
    total_count = sum(prior.values())
    for label in prior:
        prior[label] /= float(total_count)
    return prior

# 先验概率
prior = get_prior(label_index)

def get_likelihood(term_document_matrix, label_index: dict, smoothing=0):
    """
    基于训练数据，计算似然概率

    似然概率是指在某个类别下，某个单词出现的概率
    P(w|S) = (S类别下w出现的次数 + smoothing) / (S类别下所有单词出现的次数 + smoothing * |V|)

    Args:
        term_document_matrix: 特征矩阵, 行是邮件, 列是单词, 值是单词在邮件中出现的次数
    Returns:
        字典, key是标签, value 1 * N的矩阵, 表示该类别下每个单词的似然概率
    """
    likelihood = {}
    # lable是标签，index是该标签对应的邮件索引数组
    for label, index in label_index.items():
        # 特征矩阵只保留lable对应的行（由index指定），保留所有列（由:指定）
        # 然后对新的矩阵按列求和（由axis=0指定），得到每个单词在该类别下出现的次数
        # 每个单词的次数加上平滑值
        # 结果是一个1 * N的矩阵，N是单词数
        likelihood[label] = term_document_matrix[index, :].sum(axis=0) + smoothing
        # 矩阵转换成二维数组，取第一个数组（也是唯一一个数组）
        likelihood[label] = np.asarray(likelihood[label])[0]
        # 计算该类别下所有单词出现的次数
        total_count = likelihood[label].sum()
        # 计算该标签下每个单词的似然概率：(S类别下w出现的次数 + smoothing) / (S类别下所有单词出现的次数 + smoothing * |V|)
        likelihood[label] = likelihood[label] / float(total_count)
    return likelihood

smoothing = 1
likelihood = get_likelihood(term_docs, label_index, smoothing)

def get_posterior(term_document_matrix, prior, likelihood):
    """
    基于先验概率和似然概率，计算后验概率

    P(S|w) = P(w|S)^fw * P(S) / P(w)
    P(NS|w) = P(w|NS)^fw * P(NS) / P(w)

    P(S|w)/P(NS|w) = P(w|S)^fw * P(S) / P(w|NS)^fw * P(NS)
    P(S|w) + P(NS|w) = 1

    所以
    P(S|w) = P(w|S)^fw * P(S) / (P(w|S)^fw * P(S) + P(w|NS)^fw * P(NS))
    使用对数变换
    P(w|S)^fw * P(S) = exp(log(P(w|S)^fw) + log(P(S))) = exp(fw * log(P(w|S)) + log(P(S)))

    Args:
        term_document_matrix: 所有待测试邮件组成的特征矩阵, 行是邮件, 列是单词, 值是单词在邮件中出现的次数
        prior: 先验概率
        likelihood: 似然概率
    Returns:
        后验概率
    """
    # 矩阵的行数，也就是邮件数量
    num_docs = term_document_matrix.shape[0]
    # 存储每个邮件的后验概率
    posteriors = []
    # 分别计算每个邮件的后验概率
    for i in range(num_docs):
        # 初始化后验概率，key是标签，value是该标签的先验概率取对数
        posterior = {label: np.log(prior_value)
          for label, prior_value in prior.items()}
        for label, likelihood_value in likelihood.items():
            # 取出第i封邮件的特征向量
            term_document_vector = term_document_matrix.getrow(i)
            #
            counts = term_document_vector.data
            indices = term_document_vector.indices
            for count, index in zip(counts, indices):
                # 先验概率取对数，所有似然概率取对数，然后相加
                posterior[label] += np.log(likelihood_value[index]) * count
        # 选取后验概率最小的值
        min_log_posterior = min(posterior.values())
        # 为了避免数值溢出，将所有后验概率减去最小值，再取指数
        for label in posterior:
            try:
                posterior[label] = np.exp(posterior[label] - min_log_posterior)
            except:
                posterior[label] = float('inf')
        sum_posterior = sum(posterior.values())
        for label in posterior:
            if posterior[label] == float('inf'):
                posterior[label] = 1.0
            else:
                posterior[label] /= sum_posterior
        posteriors.append(posterior.copy())
    return posteriors



e_mails_test = [
'''Subject: flat screens
hello ,
please call or contact regarding the other flat screens
requested .
trisha tlapek - eb 3132 b
michael sergeev - eb 3132 a
also the sun blocker that was taken away from eb 3131 a .
trisha should two monitors also michael .
thanks
kevin moore''',
'''Subject: having problems in bed ? we can help !
cialis allows men to enjoy a fully normal sex life without
having to plan the sexual act .
if we let things terrify us, life will not be worth living
brevity is the soul of lingerie .
suspicion always haunts the guilty mind .''',
]

# 数据清洗
cleaned_test = clean_text(e_mails_test)
# 转换成特征矩阵
term_docs_test = cv.transform(cleaned_test)
# 计算后验概率
posterior = get_posterior(term_docs_test, prior, likelihood)
print(posterior)