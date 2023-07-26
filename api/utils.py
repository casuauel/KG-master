import json
import logging
import os.path
import torch
from sklearn.metrics import confusion_matrix, classification_report
import time

count_error = 0


# def build_dataset(config):
#     if os.path.exists(config.dataset_pkl):
#         dataset = pkl.load(open(config.dataset_pkl, 'rb'))
#         train = dataset['train']
#         dev = dataset['dev']
#         test = dataset['test']
#     else:
#         train_texts, train_labels = read_data(config.train_path)
#         train = BertDataset(train_texts, train_labels, config)
#         dev_texts, dev_labels = read_data(config.dev_path)
#         dev = BertDataset(dev_texts, dev_labels, config)
#         test_texts, test_labels = read_data(config.test_path)
#         test = BertDataset(test_texts, test_labels, config)
#         dataset = {'train': train, 'dev': dev, 'test': test}
#         pkl.dump(dataset, open(config.dataset_pkl, 'wb'))
#     return train, dev, test


def read_data(data_path):
    """
    接受一个数据文件的路径作为参数，从文件中读取数据集，
    并将数据集中的句子和关系标签分别存储在`texts`和
    `labels`列表中，然后返回这两个列表。
    """
    with open(data_path, 'r', encoding='utf-8') as file:
        text = file.read()
        dataset = json.loads(text)
        texts = []
        labels = []
        for data in dataset:
            labels.append(data["relation"])
            texts.append(data["sentence"])
    return texts, labels


def get_logger(log_file):
    """
    定义日志方法
    :param log_file:
    :return:
    """
    # 创建一个logging的实例 logger
    logger = logging.getLogger(log_file)
    # 设置logger的全局日志级别为Debug
    logger.setLevel(logging.DEBUG)
    # 创建一个日志文件的handler,并设置日志级别为debug
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    # 创建一个屏幕的handler,并设置日志级别为Debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # 设计日志格式
    formatter = logging.Formatter("%(asctime)s -%(name)s - %(levelname)s- %(message)s")
    # add formatter to ch and fh
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

# 用于获取当前时间的字符串表示
def get_str_time():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))

"""
    函数`get_path`会根据给定的目录路径、文件后缀
    和当前时间生成一个带有时间戳的文件路径。如果文
    件路径不存在，则会创建一个空文件，并返回文件路径
"""
def get_path(dic, suffix):
    time_str = get_str_time()
    path = r"%s/%s.%s" % (dic, time_str, suffix)
    if not os.path.exists(path):
        f = open(path, "x")
        f.close()
    return path

"""
    从给定的文件中读取命名实体识别的数据。它将每个文本
    和对应的标签存储在`all_text`和`all_label`列表
    中，并返回这两个列表。
"""
def read_ner_data(filename):
    global count_error
    with open(filename, 'r', encoding='utf8') as f:
        all_data = f.read().split('\n')
    all_text = []
    all_label = []
    text = []
    labels = []
    # count_emp = 0
    for data in all_data:
        # if count_emp == 1:
        #     count_emp = 0
        #     continue
        if data == '' and text != []:
            all_text.append(text)
            all_label.append(labels)
            text = []
            labels = []
            # count_emp += 1
        else:
            try:
                t, l = data.split(' ')
                text.append(t)
                labels.append(l)
            except Exception:
                count_error = count_error + 1
                print(data)
                continue
    print(f"count_error:{count_error}")
    return all_text, all_label


def read_data_ner(file_name):
    global count_error
    with open(file_name, 'r', encoding='utf8') as f:
        all_data = f.read().split('\n')
    all_text = []
    all_label = []
    text = []
    labels = []
    for data in all_data:
        if data == '' and text != []:
            all_text.append(text)
            all_label.append(labels)
            text = []
            labels = []
        else:
            try:
                t, l = data.split(' ')
                # 将变量`l`转换为字符串类型，并使用"-"作为分隔符进行分割，然后取分割后的结果的第一个部分
                l = str(l).split("-")[0]
                text.append(t)
                labels.append(l)
            except Exception:
                count_error = count_error + 1
                print(data)
                continue
    print(f"count_error:{count_error}")
    return all_text, all_label

# 接受真实标签和预测标签作为输入，然后计算并打印混淆矩阵和分类报告
def print_confusion_matrix(y_test, y_pred):
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

"""
    函数`get_mask`会根据输入的索引序列和最大序列长度
    生成一个掩码，用于指示哪些位置是有效的（1）或无效的（0）
"""
def get_mask(text_idx, max_len):
    mask = [1] * (len(text_idx) + 2)
    length = min(max_len, len(text_idx) + 2)
    mask = mask[:length]
    if length < max_len:
        for i in range(length, max_len):
            mask.append(0)
    return torch.tensor(mask)

# 根据输入的索引序列生成一个掩码，用于在预测时指示哪些位置是有效的（1）
def predict_mask(text_idx):
    mask = [1] * (len(text_idx) + 2)
    return torch.tensor(mask)

"""
    从给定的文件中读取语义关系数据。它将每个句子文本
    和对应的关系标签存储在`texts`和`labels`列表中，并返回这两个列表
"""
def read_sem_data(file_path):
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = json.loads(line.strip())
            labels.append(line["relation"])
            texts.append(line["sentence"])

    return texts, labels


if __name__ == '__main__':
    p = get_path("./data/log", 'log')
    print(p)
