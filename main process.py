from collections import defaultdict
import random
import pandas as pd
import csv
import numpy as np


class WindowsScatter:
    def __init__(self):
        self.all_rules_fail = 0
        self.first_rules_fail = 0
        self.second_rules_fail = 0
        self.third_rules_fail = 0

    @staticmethod
    def data_create(num):
        creator_id = []
        item_id = []
        music_id = []
        for i in range(1, 1001):
            creator_id.append("creator_id" + str(i))
        for i in range(1, 31):
            item_id.append("item_id" + str(i))
        for i in range(1, 101):
            music_id.append("music_id" + str(i))
        random.shuffle(creator_id)
        random.shuffle(item_id)
        random.shuffle(music_id)
        video_list = {}

        for j in range(num * 20):
            video_list[j] = [creator_id[random.randint(0, 999)], item_id[random.randint(0, 29)],
                             music_id[random.randint(0, 99)]]
        #video_list_frame = pd.DataFrame(video_list).T
        #video_list_frame.to_csv('video_list.csv')
        return video_list

    def scatter(self, video_dict):
        # 在长度为8窗口范围内搜索
        win_len = 8
        begin, end = 0, 7
        # 定义三个字典 用于记录视频特征出现次数，分别为：作者，类别，音乐
        map1, map2, map3 = defaultdict(int), defaultdict(int), defaultdict(int)
        # 这个i用于当窗口不符合要求时，连续向窗口最右端搜索的指针
        i = 1
        while end < 20:
            # 定义一个指针 从窗口最左端开始逐个记录
            for pointer in range(win_len):
                map1[video_dict[pointer + begin][0]] += 1  # 作者id记录
                map2[video_dict[pointer + begin][1]] += 1  # 类别id记录
                map3[video_dict[pointer + begin][2]] += 1  # 音乐id记录
                # 每记录一个视频判断一次最大值
                prod_num = max(map1.values())
                item_num = max(map2.values())
                music_num = max(map3.values())
                # 若三项中的某一个项最大值超过规定
                while prod_num > 2 or item_num > 3 or music_num > 1:
                    # 右端窗口已经到达队列末尾还出现不符合的情况，必定不可能再满足要求了
                    if end == 19:
                        self.all_rules_fail += 1
                        if prod_num > 2:
                            self.first_rules_fail += 1
                        elif item_num > 3:
                            self.second_rules_fail += 1
                        elif music_num > 1:
                            print(video_dict)
                            self.third_rules_fail += 1
                        return
                    # 如果搜索完后面的序列没有一个满足的，记录本次
                    elif end + i >= 20:
                        self.all_rules_fail += 1
                        if prod_num > 2:
                            self.first_rules_fail += 1
                        elif item_num > 3:
                            self.second_rules_fail += 1
                        elif music_num > 1:
                            self.third_rules_fail += 1
                        return
                    # 将字典中记录的先还原
                    map1[video_dict[pointer + begin][0]] -= 1  # 作者id记录
                    map2[video_dict[pointer + begin][1]] -= 1  # 类别id记录
                    map3[video_dict[pointer + begin][2]] -= 1  # 音乐id记录
                    # 把当前导致最大值超过的视频与窗口最右端后一位视频交换
                    video_dict[end + i], video_dict[pointer + begin] = video_dict[pointer + begin], video_dict[end + i]
                    # 重新在字典记录交换后的结果
                    map1[video_dict[pointer + begin][0]] += 1  # 作者id记录
                    map2[video_dict[pointer + begin][1]] += 1  # 类别id记录
                    map3[video_dict[pointer + begin][2]] += 1  # 音乐id记录
                    # 更新最大值
                    prod_num = max(map1.values())
                    item_num = max(map2.values())
                    music_num = max(map3.values())
                    # 如果最右端后一位也不符合要求，继续项后搜索
                    i += 1

                # 找到符合了交换了,还原指针
                i = 1
            # 当前窗口8格都符合要求后，窗口整体右移并且清除之前窗口的记录
            begin += 1
            end += 1
            map1, map2, map3 = defaultdict(int), defaultdict(int), defaultdict(int)
        return video_dict

    def printnum(self):
        print("all_fail_num:", self.all_rules_fail, "fail_rate:", self.all_rules_fail / 10000)
        print("first_fail_num:", self.first_rules_fail, "fail_rate:", self.first_rules_fail / 10000)
        print("second_fail_num:", self.second_rules_fail, "fail_rate:", self.second_rules_fail / 10000)
        print("third_fail_num:", self.third_rules_fail, "fail_rate:", self.third_rules_fail / 10000)


# 调试用列表
data_video = {0: ['creator_id211', 'item_id5', 'music_id5'],
              1: ['creator_id389', 'item_id27', 'music_id44'],
              2: ['creator_id233', 'item_id27', 'music_id36'],
              3: ['creator_id996', 'item_id19', 'music_id3'],
              4: ['creator_id845', 'item_id25', 'music_id78'],
              5: ['creator_id488', 'item_id20', 'music_id33'],
              6: ['creator_id738', 'item_id20', 'music_id38'],
              7: ['creator_id883', 'item_id17', 'music_id11'],
              8: ['creator_id190', 'item_id16', 'music_id5'],
              9: ['creator_id940', 'item_id6', 'music_id23'],
              10: ['creator_id678', 'item_id5', 'music_id29'],
              11: ['creator_id264', 'item_id12', 'music_id3'],
              12: ['creator_id657', 'item_id17', 'music_id21'],
              13: ['creator_id705', 'item_id12', 'music_id14'],
              14: ['creator_id269', 'item_id8', 'music_id88'],
              15: ['creator_id444', 'item_id9', 'music_id26'],
              16: ['creator_id236', 'item_id15', 'music_id15'],
              17: ['creator_id801', 'item_id30', 'music_id93'],
              18: ['creator_id245', 'item_id27', 'music_id47'],
              19: ['creator_id558', 'item_id2', 'music_id93']}

my_scatter = WindowsScatter()

#raw_video_list = pd.read_csv('video_list.csv', header=None, index_col=0, squeeze=False).T.to_dict()
#print(raw_video_list)
for i in range(1, 10001):
    if i % 1000 == 0:
        print("process_num:", i)
    data = my_scatter.data_create(1)
    #print(data)
    ans = my_scatter.scatter(data)
    #print(ans)
my_scatter.printnum()


'''print(data_video)
ans = my_scatter.scatter(data_video)
#print(ans)
my_scatter.printnum()'''

'''if __name__ == "__main__":
    ans = 1
    list1 = []
    for i in range(20):
        ans = ans * (100-i)
        list1.append(100 - i)
    #ans = ans * 19
    ans = ans / (100 ** 20)
    print(ans)
    print(list1)'''

