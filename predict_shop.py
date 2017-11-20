# -*- coding: utf-8 -*-

__version__ = 1.0
__author__ = "Yi Shixiang <yishixiang@gmail.com>"
__date__ = "2017-11-9"

import pandas as pd
import math
import numpy as np
from datetime import datetime
import random


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
        self.sampleData = None
        print("Initiation is complete\n")
        self.strategy = 'top_n'
    pass

    # #计算距离
    def dis(self, loca1, loca2):
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
    def __get_shop__(self, *mall):
        selected_shop = self.shop_info[self.shop_info['mall_id'].isin(list(mall))]
        return selected_shop

    # 返回top-N的dataframe
    def __top_N__(self, topn=None, score=None, select_shop=None, shop_id=None):
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
                    if single_wifi[2] == 'true':
                        count = count + 1
                        score[i] = score[i] + 4
                        # score[i] = score[i] + (100 - abs(float(shop_wifi[locate[0][0]][1]) -
                        #                                  float(single_wifi[1]))) * 1.5
                    else:
                        count = count + 1
                        score[i] = score[i] + 1
                        # score[i] = score[i] + 100 - abs(float(shop_wifi[locate[0][0]][1]) - float(single_wifi[1]))
        # return (score - score.min() + 1) / (score.ptp() + 1) * 100
        return score

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

    def cal_dis(self, select_info, loc):
        score = np.zeros(select_info.shape[0])
        for i in range(select_info.shape[0]):
            score[i] = self.dis((select_info.iloc[i]['longitude'], select_info.iloc[i]['latitude']), loc)
        score = score.max() - score
        return (score - score.min()) / (score.ptp()) * 100

    def __get_max_redundancy__(self, train_data):
        # Random select some data to train
        train_dataset = train_data
        temp_topn_wifi = []
        temp_topn_time = []
        temp_topn_dis = []
        for i in range(train_dataset.shape[0]):
            selected_shop = self.__get_shop__(self.shop_info[self.shop_info['shop_id'] ==
                                                             train_dataset.iloc[i]['shop_id']]['mall_id'].iloc[0])
            wifi_score = self.cal_wifi(selected_shop, train_dataset.iloc[i]['wifi_infos'])
            time_info = datetime.strptime(train_dataset.iloc[i]['time_stamp'], '%Y-%m-%d %H:%M')
            time_score = self.cal_time(selected_shop, time_info.hour * 60 + time_info.second)
            dis_score = self.cal_dis(selected_shop,
                                     (train_dataset.iloc[i]['longitude'], train_dataset.iloc[i]['latitude']))
            temp_topn_wifi.append(self.__top_N__(select_shop=selected_shop,
                                                 shop_id=train_dataset.iloc[i]['shop_id'],
                                                 score=wifi_score))
            temp_topn_time.append(self.__top_N__(select_shop=selected_shop,
                                                 shop_id=train_dataset.iloc[i]['shop_id'],
                                                 score=time_score))
            temp_topn_dis.append(self.__top_N__(select_shop=selected_shop,
                                                shop_id=train_dataset.iloc[i]['shop_id'],
                                                score=dis_score))
        temp_topn_wifi = np.array(temp_topn_wifi)
        temp_topn_time = np.array(temp_topn_time)
        temp_topn_dis = np.array(temp_topn_dis)
        self.parameter['top_n_time'] = temp_topn_time.max()
        self.parameter['top_n_wifi'] = temp_topn_wifi.max()
        self.parameter['top_n_dis'] = temp_topn_dis.max()
        self.parameter['weight_wifi'] = np.average(temp_topn_wifi)
        self.parameter['weight_time'] = np.average(temp_topn_time)
        self.parameter['weight_dis'] = np.average(temp_topn_dis)
        s = np.average(temp_topn_wifi) + np.average(temp_topn_time) + np.average(temp_topn_dis)
        self.parameter['weight_wifi'] = (s - np.average(temp_topn_wifi)) / (s * 2)
        self.parameter['weight_time'] = (s - np.average(temp_topn_time)) / (s * 2)
        self.parameter['weight_dis'] = (s - np.average(temp_topn_dis)) / (s * 2)

        print("Get the max redundancy\n")
        pass

    def __predict_shop__(self, selected_shop, data, weight_wifi, weight_time, weight_dis):
        wifi_score = self.cal_wifi(selected_shop, data['wifi_infos'])
        time_info = datetime.strptime(data['time_stamp'], '%Y-%m-%d %H:%M')
        time_score = self.cal_time(selected_shop, time_info.hour * 60 + time_info.second)
        dis_score = self.cal_dis(selected_shop, (data['longitude'], data['latitude']))
        total_score = (weight_wifi * wifi_score +
                       weight_time * time_score +
                       weight_dis * dis_score)
        return selected_shop.iloc[total_score.argmax()]['shop_id']

    #########################################
    #   Grid Search
    #
    #########################################
    def __grid_dots__(self):
        self.parameter['weight_wifi']
        self.parameter['weight_time']
        # self.parameter['weight_dis']
        grid_wifi = np.arange(self.parameter['weight_wifi'], 1, 0.02)
        grid_time = np.linspace(self.parameter['weight_time'], 0, grid_wifi.shape[0])
        grid_dis = np.linspace(self.parameter['weight_dis'], 0, grid_wifi.shape[0])
        c = np.column_stack((grid_wifi, gri d_time, grid_dis))
        return c

    def __grid_search__(self, train_dataset):
        grid = self.__grid_dots__()
        acc_array = []
        print("Grid searching... ...\n")
        for select_dot in grid:
            j = 0
            for i in range(train_dataset.shape[0]):
                selected_shop = self.__get_shop__(self.shop_info[self.shop_info['shop_id'] ==
                                                                 train_dataset.iloc[i]['shop_id']]['mall_id'].iloc[0])
                p_shop = self.__predict_shop__(selected_shop, train_dataset.iloc[i],
                                               select_dot[0], select_dot[1], select_dot[2])
                if p_shop == train_dataset.iloc[i]['shop_id']:
                    j = j + 1
            acc_array.append(j / train_dataset.shape[0])
        acc_array = np.array(acc_array)
        index_grid = np.argmax(acc_array)
        self.parameter['weight_wifi'] = grid[index_grid, 0]
        self.parameter['weight_time'] = grid[index_grid, 1]
        print("Grid complete\n")

    #########################################
    #   Genetic Algorithm
    #
    #########################################
    def __variation__(self, next_generation, varitation_ratio):
        x = np.random.random(next_generation.shape[1])
        for i in range(x.shape[0]):
            if x[i] < varitation_ratio:
                next_generation[i][0] = next_generation[i][0] + random.uniform(-0.3, 0.3)
                next_generation[i][1] = next_generation[i][1] + random.uniform(-0.3, 0.3)
                next_generation[i][2] = next_generation[i][2] + random.uniform(-0.3, 0.3)
        return next_generation
        pass

    def __cross__(self, next_generation, cross_ratio):
        x = np.random.random(next_generation.shape[1])
        for i in range(x.shape[0]):
            if x[i] < cross_ratio:
                next_generation[i][0] = next_generation[i][0] + next_generation[i + 1][0] * random.uniform(-0.5, 0.5)
                next_generation[i][1] = next_generation[i][1] + next_generation[i + 1][1] * random.uniform(-0.5, 0.5)
                next_generation[i][2] = next_generation[i][2] + next_generation[i + 1][2] * random.uniform(-0.5, 0.5)
        return next_generation

    def __wheel_selection__(self, genetic, fitness):
        s = fitness / fitness.sum()
        s = np.cumsum(s)
        result = np.zeros((genetic.shape[0], 3))
        select_proba = np.random.random(s.shape[0])
        for i in range(s.shape[0]):
            x = np.where(s > select_proba[i])[0].min()
            result[i] = genetic[x]
        return result

    def __fitness__(self, genetic, train_dataset):
        j = 0
        fit = np.zeros(genetic.shape[0])
        for x in range(genetic.shape[0]):
            for i in range(train_dataset.shape[0]):
                selected_shop = self.__get_shop__(self.shop_info[self.shop_info['shop_id'] ==
                                                                 train_dataset.iloc[i]['shop_id']]['mall_id'].iloc[0])
                wifi_score = self.cal_wifi(selected_shop, train_dataset.iloc[i]['wifi_infos'])
                time_info = datetime.strptime(train_dataset.iloc[i]['time_stamp'], '%Y-%m-%d %H:%M')
                time_score = self.cal_time(selected_shop, time_info.hour * 60 + time_info.second)
                dis_score = self.cal_dis(selected_shop, (train_dataset.iloc[i]['longitude'], train_dataset.iloc[i]['latitude']))
                total_score = (genetic[x][0] * wifi_score +
                               genetic[x][1] * time_score +
                               genetic[x][2] * dis_score)
                if selected_shop.iloc[total_score.argmax()]['shop_id'] == train_dataset.iloc[i]['shop_id']:
                    j += 15
                else:
                    j += random.randint(1, 4)
            # if j != 0:
            #     total_score = 100
            # else:
            #     total_score = random.randint(10, 70)
            fit[x] = j
        return fit

    def __genetic_search__(self, train_dataset, pro_var, pro_cross):
        ###############
        # Initiate seed
        ###############
        seed_wifi = np.random.triangular(self.parameter['weight_wifi'], (1 + self.parameter['weight_wifi']) / 2, 1, 20)
        seed_time = np.random.triangular(0, self.parameter['weight_time'] / 2, self.parameter['weight_time'], 20)
        seed_dis = np.random.triangular(0, self.parameter['weight_dis'] / 2, self.parameter['weight_dis'], 20)
        seed = np.column_stack((seed_wifi, seed_time, seed_dis))
        fit = np.array(self.__fitness__(seed, train_dataset))
        print(seed.shape)
        for times in range(20):
            print("Time:", times)
            next_generation = self.__wheel_selection__(seed, fit)
            print(next_generation)
            next_generation = self.__cross__(next_generation, pro_cross)
            next_generation = self.__variation__(next_generation, pro_var)
            fit = np.array(self.__fitness__(next_generation, train_dataset))
        self.parameter['weight_wifi'] = next_generation[0][np.argmax(fit)]
        self.parameter['weight_time'] = next_generation[1][np.argmax(fit)]
        self.parameter['weight_dis'] = next_generation[2][np.argmax(fit)]
        print("Max fitness:\t", fit.max())

    def test(self, fraction):
        test_dataset = self.train_info.sample(frac=fraction)
        j = 0
        for i in range(test_dataset.shape[0]):
            selected_shop = self.__get_shop__(self.shop_info[self.shop_info['shop_id'] ==
                                                             test_dataset.iloc[i]['shop_id']]['mall_id'].iloc[0])
            wifi_score = self.cal_wifi(selected_shop, test_dataset.iloc[i]['wifi_infos'])
            time_info = datetime.strptime(test_dataset.iloc[i]['time_stamp'], '%Y-%m-%d %H:%M')
            time_score = self.cal_time(selected_shop, time_info.hour * 60 + time_info.second)
            dis_score = self.cal_dis(selected_shop, (test_dataset.iloc[i]['longitude'],
                                                     test_dataset.iloc[i]['latitude']))
            total_score = (self.parameter['weight_wifi'] * wifi_score +
                           self.parameter['weight_time'] * time_score +
                           self.parameter['weight_dis'] * dis_score)
            print(i)
            if selected_shop.iloc[total_score.argmax()]['shop_id'] == test_dataset.iloc[i]['shop_id']:
                j = j + 1
        print("ACC:", j / test_dataset.shape[0])

    def train(self, fraction, strategy='grid_search'):
        ##############################################
        # Optimize top_n_wifi, top_n_time, top_n_dis #
        ##############################################
        if strategy == 'grid_search':
            self.strategy = 'grid_search'
            train_dataset = self.train_info.sample(frac=fraction)
            self.__get_max_redundancy__(train_data=train_dataset)
            print("Parameter Seed:\n", self.parameter)
            self.__grid_search__(train_dataset=train_dataset)
            print("Trained parameter:\n", self.parameter)
        elif strategy == "genetic":
            pro_var = 0.06
            pro_cross = 0.6
            train_dataset = self.train_info.sample(frac=fraction)
            self.__get_max_redundancy__(train_data=train_dataset)
            print("Parameter Seed:\n", self.parameter)
            self.__genetic_search__(train_dataset, pro_var, pro_cross)
            print("Trained parameter:\n", self.parameter)
            pass
        else:
            pass
        pass

    def predict(self):
        predict_shop = []
        print("Predicting... ...")
        # for i in range(self.predict_info.shape[0]):
        for i in range(320000, 400000):
            selected_shop = self.__get_shop__(self.predict_info.iloc[i]['mall_id'])
            wifi_score = self.cal_wifi(selected_shop, self.predict_info.iloc[i]['wifi_infos'])
            time_info = datetime.strptime(self.predict_info.iloc[i]['time_stamp'], '%Y-%m-%d %H:%M')
            time_score = self.cal_time(selected_shop, time_info.hour * 60 + time_info.second)
            dis_score = self.cal_dis(selected_shop, (self.predict_info.iloc[i]['longitude'],
                                                     self.predict_info.iloc[i]['latitude']))
            total_score = (self.parameter['weight_wifi'] * wifi_score +
                           self.parameter['weight_time'] * time_score +
                           self.parameter['weight_dis'] * dis_score)
            predict_shop.append(selected_shop.iloc[total_score.argmax()]['shop_id'])
            print(i)
        result = pd.DataFrame({'row_id': self.predict_info.row_id, 'shop_id': predict_shop})
        result.fillna('s_666').to_csv('first.csv', index=None)
        print("Done!")


if __name__ == '__main__':
    example = predict_shop(predict_info='evaluation_public.csv',
                           shop_info='shop_info.csv',
                           train_info='train-ccf_first_round_user_shop_behavior.csv'
                           )

    example.train(fraction=0.001, strategy="genetic")
    example.test(fraction=0.001)
    # example.predict()
    pass
