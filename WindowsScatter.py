"""
----------------
author:Joey
date:2022.5.27
function:Use rolling window to scatter video sequences according to features
----------------
"""

from collections import defaultdict
import random
import time
import pandas as pd
import numpy as np


# 随机生成长度为num的视频序列,包含三个特征
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
    for j in range(num):
        video_list[j] = [creator_id[random.randint(0, 999)], item_id[random.randint(0, 29)],
                         music_id[random.randint(0, 99)]]
    return video_list


class WindowsScatter:
    def __init__(self, video_dict, windows_len):
        # 滚动窗口的长度
        self.windows_len = windows_len
        # 一个视频序列的长度
        self.video_dict_len = len(video_dict)
        # 打散各项规则失败次数记录
        self.all_rules_fail = 0
        self.first_rules_fail = 0
        self.second_rules_fail = 0
        self.third_rules_fail = 0

    def positive_order_scatter(self, video_dict):
        # 在长度为8窗口范围内搜索
        begin, end = 0, self.windows_len - 1
        # 定义三个字典 用于记录视频特征出现次数，分别为：作者，类别，音乐
        map1, map2, map3 = defaultdict(int), defaultdict(int), defaultdict(int)
        # 这个i用于当窗口不符合要求时，连续向窗口最右端搜索的指针
        i = 1
        while end < len(video_dict):
            # 定义一个指针 从窗口最左端开始逐个记录
            for pointer in range(self.windows_len):
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
                    if end == self.video_dict_len - 1:
                        self.all_rules_fail += 1
                        if prod_num > 2:
                            self.first_rules_fail += 1
                        elif item_num > 3:
                            self.second_rules_fail += 1
                        elif music_num > 1:
                            # print(video_dict)
                            self.third_rules_fail += 1
                        return
                    # 如果搜索完后面的序列没有一个满足的，记录本次
                    elif end + i > self.video_dict_len - 1:
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

    def reverse_order_scatter(self, video_dict):
        # 在长度为8窗口范围内搜索,倒序，从序列末尾开始，窗口反向，右端为头，左端为尾
        begin, end = self.video_dict_len - 1, self.video_dict_len - self.windows_len
        # 定义三个字典 用于记录视频特征出现次数，分别为：作者，类别，音乐
        map1, map2, map3 = defaultdict(int), defaultdict(int), defaultdict(int)
        # 这个i用于当窗口不符合要求时，连续向窗口最左端搜索的指针
        i = 1
        while end >= 0:
            # 定义一个指针 从窗口最左端开始逐个记录
            for pointer in range(self.windows_len):
                map1[video_dict[begin - pointer][0]] += 1  # 作者id记录
                map2[video_dict[begin - pointer][1]] += 1  # 类别id记录
                map3[video_dict[begin - pointer][2]] += 1  # 音乐id记录
                # 每记录一个视频判断一次最大值
                prod_num = max(map1.values())
                item_num = max(map2.values())
                music_num = max(map3.values())
                # 若三项中的某一个项最大值超过规定
                while prod_num > 2 or item_num > 3 or music_num > 1:
                    # 之前已对序列进行过正序打散，在第二次倒序打散的情况中依旧不符合规定要求，判定为打散失败
                    if end == 0:
                        #print(video_dict)
                        self.all_rules_fail += 1
                        if prod_num > 2:
                            self.first_rules_fail += 1
                        elif item_num > 3:
                            self.second_rules_fail += 1
                        elif music_num > 1:
                            self.third_rules_fail += 1
                        return video_dict
                    # 如果搜索完后面的序列没有一个满足的，记录本次
                    elif end - i < 0:
                        #print(video_dict)
                        self.all_rules_fail += 1
                        if prod_num > 2:
                            self.first_rules_fail += 1
                        elif item_num > 3:
                            self.second_rules_fail += 1
                        elif music_num > 1:
                            self.third_rules_fail += 1
                        return video_dict
                    # 将字典中记录的先还原
                    map1[video_dict[begin - pointer][0]] -= 1  # 作者id记录
                    map2[video_dict[begin - pointer][1]] -= 1  # 类别id记录
                    map3[video_dict[begin - pointer][2]] -= 1  # 音乐id记录
                    # 把当前导致最大值超过的视频与窗口最左端前一位视频交换
                    video_dict[end - i], video_dict[begin - pointer] = video_dict[begin - pointer], video_dict[end - i]
                    # 重新在字典记录交换后的结果
                    map1[video_dict[begin - pointer][0]] += 1  # 作者id记录
                    map2[video_dict[begin - pointer][1]] += 1  # 类别id记录
                    map3[video_dict[begin - pointer][2]] += 1  # 音乐id记录
                    # 更新最大值
                    prod_num = max(map1.values())
                    item_num = max(map2.values())
                    music_num = max(map3.values())
                    # 如果最左端前一位也不符合要求，继续向前搜索
                    i += 1

                # 找到符合了交换了,还原指针
                i = 1
            # 当前窗口8格都符合要求后，窗口整体左移并且清除之前窗口的记录
            begin -= 1
            end -= 1
            map1, map2, map3 = defaultdict(int), defaultdict(int), defaultdict(int)
        return video_dict

    # v1.0.0算法，只使用一次正序打散
    def scatter_v1_0_0(self, video_dict):
        self.positive_order_scatter(video_dict)

    # v2.0.0算法，当一次正序打散失败，则对那次正序打散失败后的序列做倒序处理，进行一次倒序打散
    def scatter_v2_0_0(self, video_dict):
        # 在长度为8窗口范围内搜索
        begin, end = 0, self.windows_len - 1
        # 定义三个字典 用于记录视频特征出现次数，分别为：作者，类别，音乐
        map1, map2, map3 = defaultdict(int), defaultdict(int), defaultdict(int)
        # 这个i用于当窗口不符合要求时，连续向窗口最右端搜索的指针
        i = 1
        while end < self.video_dict_len:
            # 定义一个指针 从窗口最左端开始逐个记录
            for pointer in range(self.windows_len):
                map1[video_dict[pointer + begin][0]] += 1  # 作者id记录
                map2[video_dict[pointer + begin][1]] += 1  # 类别id记录
                map3[video_dict[pointer + begin][2]] += 1  # 音乐id记录
                # 每记录一个视频判断一次最大值
                prod_num = max(map1.values())
                item_num = max(map2.values())
                music_num = max(map3.values())
                # 若三项中的某一个项最大值超过规定
                while prod_num > 2 or item_num > 3 or music_num > 1:

                    if end == self.video_dict_len - 1:
                        # 正序打散失败，进行倒序打散
                        self.reverse_order_scatter(video_dict)
                        return video_dict
                    elif end + i > self.video_dict_len - 1:
                        # 正序打散失败，进行倒序打散
                        self.reverse_order_scatter(video_dict)
                        return video_dict
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


'''
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
print(data_video)
ans = my_scatter.scatter_v2_0_0(data_video)
print(ans)
my_scatter.printnum()'''

if __name__ == "__main__":
    # 生产10000个的视频序列
    total_video_list = []
    for _ in range(10000):
        # 自定义序列长度
        total_video_list.append(data_create(40))

    # 初始化对象,传入第一个序列是为了获取每个视频序列的长度
    my_scatter = WindowsScatter(total_video_list[0], 16)

    time_start = time.time()
    for i in range(10000):
        if (i + 1) % 1000 == 0:
            print("process_num:", i + 1)
        my_scatter.scatter_v2_0_0(total_video_list[i])
    my_scatter.printnum()
    time_end = time.time()
    print("use time:", (time_end - time_start))

