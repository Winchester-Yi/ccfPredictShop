# -*- coding: utf-8 -*-

__version__ = 0.9
__author__ = "Yi Shixiang <yishixiang@gmail.com>"
__date__ = "2017-11-9"

import pandas as pd
import math
import numpy as np
from datetime import datetime


class predict_shop(object):
    """docstring for predict_shop"""

    def __init__(self, predict_info, shop_info, train_info=None, parameter={'top_n_wifi': 0,
                                                                            'top_n_time': 0,
                                                                            'top_n_dis': 0,
                                                                            'weight_wifi': 0.0,
                                                                            'weight_time': 0.0,
                                                                            'weight_dis': 0.0}):
        super(predict_shop, self).__init__()
        if predict_info is not None:
            self.predict_info = pd.read_csv(str(predict_info))
        else:
            print("There is no predict_info file\n")
        if shop_info is not None:
            self.shop_info = pd.read_csv(str(shop_info))
        else:
            print("There is no shop_info file\n")
        if train_info is not None:
            self.train_info = pd.read_csv(str(train_info))
        else:
            print("There is no train file\n")
        self.parameter = parameter
        print("Initiation is complete\n")
    pass

    # #计算距离
    def dis(loca1, loca2):
        lo1 = loca1[0]
        la1 = loca1[1]
        lo2 = loca2[0]
        la2 = loca2[1]
        radlo1 = lo1 * 0.01745329252
        radla1 = la1 * 0.01745329252
        radlo2 = lo2 * 0.01745329252
        radla2 = la2 * 0.01745329252
        s = 2 * math.asin(math.sqrt(math.pow(math.sin((radla1 - radla2) / 2), 2) +
                                    math.cos(radla1) * math.cos(radla2) * (
            math.pow(math.sin((radlo1 - radlo2) / 2), 2))))
        return s

    # #寻找对应商场的店铺
    # shop_info: user_shop_behavior.csv + shop_info.csv
    # mall: 预测商铺所在商场的名称
    def get_shop(self, *mall):
        selected_shop = self.shop_info[self.shop_info['mall_id'].isin(list(mall))]
        return selected_shop

    # 返回top-N的dataframe
    def top_N(self, topn=None, score=None, select_shop=None, shop_id=None):
        if shop_id is None:
            index_array = np.argsort(score)
            # #sort self.shop_info by key and select top-N
            return select_shop.iloc[index_array[-1:-1 - topn:-1]]["shop_id"]
        else:
            a = select_shop.as_matrix(['shop_id']).reshape(select_shop.shape[0])
            b = np.argsort(score)
            # #sort self.shop_info by key and return the order of the shop_id
            return int(score.shape[0] - np.where(b == np.where(a == shop_id)[0])[0][0])

    # 计算wifi信息匹配度
    # Input
    # select_info: selected shop info
    # wifi_info: 测试集中预测客户的wifi_info
    # Output
    # score: wifi匹配度得分, 分值越高, 匹配度越高
    def cal_wifi(self, select_info, wifi_info):
        score = np.zeros(select_info.shape[0])
        shop_wifi = []
        user_wifi = [x.split('|') for x in wifi_info.split(';')]
        for i in range(select_info.shape[0]):
            count = 0
            shop_wifi = np.array([x.split('|') for x in select_info.iloc[i]['wifi_infos'].split(';')])
            for single_wifi in user_wifi:
                locate = np.where(np.isin(shop_wifi, single_wifi[0]))
                if(locate[0].shape[0] != 0):
                    count = count + 1
                    score[i] = score[i] + 100 - abs(float(shop_wifi[locate[0][0]][1]) - float(single_wifi[1]))
        return (score - score.min()) / (score.ptp()) * 100

    # 计算time信息相似度
    # Input
    # select_info: selected shop info
    # time_info: 测试集中客户的time_info
    # Output
    # score: wifi匹配度得分, 分值越高, 匹配度越高
    def cal_time(self, select_info, time_info):
        score = np.zeros(select_info.shape[0])
        shop_time = []
        for i in range(select_info.shape[0]):
            shop_time = np.array([float(x) for x in select_info.iloc[i]['time_stamp'].split(';')])
            score[i] = np.abs(shop_time - time_info).min()
        return (score - score.min()) / (score.ptp()) * 100

    def get_max_redundancy(self, fraction):
        # Random select some data to train
        train_dataset = self.train_info.sample(frac=fraction)
        temp_topn_wifi = []
        temp_topn_time = []
        for i in range(train_dataset.shape[0]):
            selected_shop = self.get_shop(self.shop_info[self.shop_info['shop_id'] ==
                                                         train_dataset.iloc[i]['shop_id']]['mall_id'].iloc[0])
            wifi_score = self.cal_wifi(selected_shop, train_dataset.iloc[i]['wifi_infos'])
            time_info = datetime.strptime(train_dataset.iloc[i]['time_stamp'], '%Y-%m-%d %H:%M')
            time_score = self.cal_time(selected_shop, time_info.hour * 60 + time_info.second)
            temp_topn_wifi.append(self.top_N(select_shop=selected_shop,
                                             shop_id=train_dataset.iloc[i]['shop_id'],
                                             score=wifi_score))
            temp_topn_time.append(self.top_N(select_shop=selected_shop,
                                             shop_id=train_dataset.iloc[i]['shop_id'],
                                             score=time_score))
            print(i)
        temp_topn_wifi = np.array(temp_topn_wifi)
        temp_topn_time = np.array(temp_topn_time)
        self.parameter['top_n_time'] = temp_topn_time.max()
        self.parameter['top_n_wifi'] = temp_topn_wifi.max()
        self.parameter['weight_wifi'] = np.average(temp_topn_wifi)
        self.parameter['weight_time'] = np.average(temp_topn_time)
        pass

    # Train Parameter
    # N-Fold train
    # Parameter:
    # top_n_wifi top_n_time top_n_dis
    # weight_wifi weight_tiem weight_dis
    def train(self, fraction):
        ##############################################
        # Optimize top_n_wifi, top_n_time, top_n_dis #
        ##############################################
        self.get_max_redundancy()
        for i in range(self.train_info.shap[0]):
            selected_shop = self.get_shop(self.train_info.iloc[i]['mall_id'])
            wifi_score = self.cal_wifi(selected_shop, self.train_info.iloc[i]['wifi_infos'])
            time_info = datetime.strptime(self.train_info.iloc[i]['time_stamp'], '%Y/%m/%d %H:%M')
            time_score = self.cal_time(selected_shop, time_info.hour * 60 + time_info.second)
            self.top_N(selected_shop, 1, wifi_score)
            self.top_N(selected_shop, 1, time_score)
        pass

    # 其余特征
    # evaluate_info: 测试集
    # shop_info: shop_id, category, longitude, latitude, price, mall_id, time_stamp, wifi_infos
    def predict(self):
        predict_shop = []
        for i in range(self.predict_info.shape[0]):
            selected_shop = self.get_shop(self.predict_info.iloc[i]['mall_id'])
            wifi_score = self.cal_wifi(selected_shop, self.predict_info.iloc[i]['wifi_infos'])
            time_info = datetime.strptime(self.predict_info.iloc[i]['time_stamp'], '%Y-%m-%d %H:%M')
            time_score = self.cal_time(selected_shop, time_info.hour * 60 + time_info.second)
            total_score = 0.71 * wifi_score + 0.29 * time_score
            predict_shop.append(selected_shop.iloc[total_score.argmax()]['shop_id'])
            print(i)
        result = pd.DataFrame({'row_id': self.predict_info.row_id, 'shop_id': predict_shop})
        result.fillna('s_666').to_csv('first.csv', index=None)

    def test(self, fraction):
        test_dataset = self.train_info.sample(frac=fraction)
        j = 0
        for i in range(test_dataset.shape[0]):
            selected_shop = self.get_shop(self.shop_info[self.shop_info['shop_id'] ==
                                                         test_dataset.iloc[i]['shop_id']]['mall_id'].iloc[0])
            wifi_score = self.cal_wifi(selected_shop, test_dataset.iloc[i]['wifi_infos'])
            time_info = datetime.strptime(test_dataset.iloc[i]['time_stamp'], '%Y-%m-%d %H:%M')
            time_score = self.cal_time(selected_shop, time_info.hour * 60 + time_info.second)
            total_score = 0.71 * wifi_score + 0.29 * time_score
            print(i)
            if selected_shop.iloc[total_score.argmax()]['shop_id'] == test_dataset.iloc[i]['shop_id']:
                j = j + 1
        print("ACC:", j / test_dataset.shape[0])
        pass

    # # 发生行为时最大距离
    # loca1 = info.as_matrix(['longitude1', 'latitude1'])
    # loca2 = info.as_matrix(['longitude2', 'latitude2'])
    # max = 0
    # for (x, y, i) in zip(loca1, loca2, range(loca1.shape[0])):
    #     distence = dis(x, y)
    #     if max <= distence:
    #         max = distence
    #     if i % 100 == 0:
    #         print("%d times" % i)
    #         print(max)
    # print(max)
    # print("max1 = 0.24418311746329538")
    # print("max2 = 0.016919575866857273")
    # print("max3 = 0.2885169098608471")


if __name__ == '__main__':
    example = predict_shop(predict_info='evaluation_public.csv',
                           shop_info='shop_info.csv',
                           train_info='train-ccf_first_round_user_shop_behavior.csv'
                           )

    example.predict()
    pass
