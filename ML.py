import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
from collections import Counter
from sklearn.naive_bayes import MultinomialNB as MNB #
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import cross_val_score
import lightgbm as lgb

#思路：采用和model_1同样的特征选择,TF-IDF做特征处理，然后利用机器学习去进行学习（LR，LGB），需要把train数据spilit为train_data,validation_data

def clean_text(origin_text):
    # 去除html标签
    text = BeautifulSoup(origin_text).get_text()
    # 去掉标点符号及非法字符
    text = re.sub("[^a-zA-Z]", " ", text)
    # 将字符全部转化为小写，并通过空格符进行分词处理
    words = text.lower().split()
    # 去掉停用词
    stop_words = set(stopwords.words("english"))
    meaningful_words = [w for w in words if w not in stop_words]
    # 将剩下的词还原成str类型
    cleaned_text = ' '.join(meaningful_words)
    return cleaned_text

def combine_information(row):
    '''
    输入一行数据，将带有文本信息的四个列合成一列
    '''
    columns_key = ['company_profile', 'description', 'requirements', 'benefits']
    outputs = ''
    for col in columns_key:
        if not pd.isnull(row[col]):
            outputs += str(row[col])
    return outputs

def TfIdf(train_df, test_df):
    print('Creating x_train, x_test, y_train……\n')
    if os.path.exists('./x_train_tfidf.pkl') and os.path.exists('./x_test_tfidf.pkl') and os.path.exists('./y_train.pkl'):
        x_train, x_test, y_train = pickle.load(open('./x_train_tfidf.pkl','rb')),pickle.load(open('./x_test_tfidf.pkl','rb')),pickle.load(open('./y_train.pkl','rb'))
    else:
        # 初始化tfidf，文本向量化，取了5000个词，然后每个词的idf(在所有文本出现的反概率)都求出来了
        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1,max_features=5000)
        # 数量向量化
        print('Creating the TFIDF vector……\n')
        # 在train上训练TF-IDF
        tfidf.fit(train_df['text'])
        # 将train中合并的信息向量化,得到了(17680, 5000)的矩阵，每个值都是对应的每个词tf-idf(该字在该文本中出现的概率*在所有文本出现的反概率)
        x_train = tfidf.transform(train_df['text'])
        x_train = x_train.toarray()
        # 将test中合并的信息向量化,得到了(198, 5000)的矩阵
        x_test = tfidf.transform(test_df['text'])
        x_test = x_test.toarray()
        # 得到y_train
        y_train = train_df['fraudulent']
        # 保存fit出来的x_train, x_test,y_train
        with open('./x_train_tfidf.pkl', 'wb') as fw:
            pickle.dump(x_train, fw)
        with open('./x_test_tfidf.pkl', 'wb') as fw:
            pickle.dump(x_test, fw)
        with open('./y_train.pkl', 'wb') as fw:
            pickle.dump(y_train, fw)
    # 打印x_train, x_test, y_train的shape信息
    print('x_train:{}'.format(x_train.shape))
    print('x_test:{}'.format(x_test.shape))
    print('y_train:{}'.format(y_train.shape))
    print('0 stands for True, 1 stands for False: {}\n'.format(Counter(y_train)))
    return x_train, x_test, y_train

def test(row):
    columns = ['company_profile','description','requirements','benefits']
    ans = ''
    for column in columns:
        if not pd.isnull(row[column]):
            ans += row[column]
    return ans

def main():
    # 打开数据
    train_df = pd.read_csv('../data/train.csv')
    test_df = pd.read_csv('../data/test.csv')
    # 合并重要信息，得到text的列
    train_df['text'] = train_df.apply(lambda row:test(row),axis = 1)
    test_df['text'] = test_df.apply(lambda row:test(row),axis = 1)
    # 对text列进行清理处理
    train_df['text'] = train_df['text'].apply(lambda x:clean_text(x))
    test_df['text'] = test_df['text'].apply(lambda x:clean_text(x))
    # 将train_df打乱，然后重置dateframe的index
    train_df = train_df.sample(frac = 1).reset_index(drop= True)
    # 得到x_train, x_test, y_train
    x_train, x_test, y_train = TfIdf(train_df, test_df)

    # 采用LR回归，先进行构建模型
    print('开始训练模型,LR……\n')
    model_lr = LR(solver = 'liblinear')
    model_lr.fit(x_train,y_train)
    # 用10折交叉验证，验证准确率
    print('10折交叉验证：Accuracy: ',end ='')
    print(np.mean(cross_val_score(model_lr, x_train, y_train, cv = 10, scoring= 'accuracy')))
    # 做预测
    preds = model_lr.predict(x_test)
    submission = pd.DataFrame({'id':range(len(preds)),'pred':preds}) # 将预测值转化为DataFrame
    submission.to_csv("./lr_submission.csv",index = False, header= False)
    print('\n预测完成，保存在./lr_submission.csv文件中。')

    # sklearn 用到了label encoder
    # 线性模型处理完之后，stark，然后在输入到lgb中
    # LGBM
    #https://blog.csdn.net/qq_34782535/article/details/89714372
    print('开始训练模型,LGBM……\n')
    train_x, valid_x, train_y, valid_y = train_test_split(x_train, y_train, test_size=0.333, random_state=0)  # 分训练集和验证集
    train = lgb.Dataset(train_x, train_y)
    valid = lgb.Dataset(valid_x, valid_y, reference=train)
    parameters = {
        'max_depth': [15, 20, 25, 30, 35],
        'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
        'feature_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
        'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
        'bagging_freq': [2, 4, 5, 6, 8],
        'lambda_l1': [0, 0.1, 0.4, 0.5, 0.6],
        'lambda_l2': [0, 10, 15, 35, 40],
        'cat_smooth': [1, 10, 15, 20, 35]
    }
    gbm = lgb.LGBMClassifier(boosting_type='gbdt',
                             objective='binary',
                             metric='auc',
                             verbose=0,
                             learning_rate=0.01,
                             num_leaves=35,
                             feature_fraction=0.8,
                             bagging_fraction=0.9,
                             bagging_freq=8,
                             lambda_l1=0.6,
                             lambda_l2=0)
    # 有了gridsearch我们便不需要fit函数
    gsearch = GridSearchCV(gbm, param_grid=parameters, scoring='accuracy', cv=3)
    gsearch.fit(train_x, train_y)

    print("Best score: %0.3f" % gsearch.best_score_)
    print("Best parameters set:")
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    # 做预测
    preds = gsearch.predict(x_test)
    submission = pd.DataFrame({'id':range(len(preds)),'pred':preds}) # 将预测值转化为DataFrame
    submission.to_csv("./lgb_submission.csv",index = False, header= False)
    print('\n预测完成，保存在./lgb_submission.csv文件中。')


if __name__ == '__main__':
    main()