import numpy as np
import math
import collections


class DPPModel(object):
    def __init__(self, **kwargs):
        self.item_count = kwargs['item_count']
        self.item_embed_size = kwargs['item_embed_size']
        self.max_iter = kwargs['max_iter']
        self.epsilon = kwargs['epsilon']

    def build_kernel_matrix(self):
        rank_score = np.random.random(size=(self.item_count))  # 用户和每个item的相关性
        item_embedding = np.random.randn(self.item_count, self.item_embed_size)  # item的embedding
        item_embedding = item_embedding / np.linalg.norm(item_embedding, axis=1, keepdims=True)
        sim_matrix = np.dot(item_embedding, item_embedding.T)  # item之间的相似度矩阵
        self.kernel_matrix = rank_score.reshape((self.item_count, 1)) \
                             * sim_matrix * rank_score.reshape((1, self.item_count))

    def dpp(self):
        c = np.zeros((self.max_iter, self.item_count))
        d = np.copy(np.diag(self.kernel_matrix))  # 一维数组形式返回核矩阵对角线元素
        j = np.argmax(d)
        Yg = [j]
        Yscore = [np.max(d)]
        iter = 0
        Z = list(range(self.item_count))
        while len(Yg) < self.max_iter:
            Z_Y = set(Z).difference(set(Yg))
            # 将其余所有的与最大的那个计算 更新d
            for i in Z_Y:
                if iter == 0:
                    ei = self.kernel_matrix[j, i] / np.sqrt(d[j])
                else:
                    ei = (self.kernel_matrix[j, i] - np.dot(c[:iter, j], c[:iter, i])) / np.sqrt(d[j])
                c[iter, i] = ei
                d[i] = d[i] - ei * ei
            d[j] = 0
            j = np.argmax(d)  # 增加item后组成了新的核矩阵对角线也就是L 获取其中最大的进行下一轮遍历
            if d[j] < self.epsilon:
                break
            Yg.append(j)
            Yscore.append(np.max(d))
            iter += 1
        Yg_Yscore = {}
        for k in range(len(Yg)):
            Yg_Yscore[Yg[k]] = Yscore[k]
        return Yg_Yscore

    def dpp_beam(self):
        c = np.zeros((self.max_iter, self.item_count))
        d = np.copy(np.diag(self.kernel_matrix))  # 一维数组形式返回核矩阵对角线元素
        j = np.argmax(d)  # 第一个取最大值 后面的每个由前三个大值综合得出
        Yg = [j]
        iter = 0

        beam_container = []
        #print(top3[0])

        while len(Yg) < self.max_iter:
            top3 = np.argsort(d)[-3:][::-1]  # 取最大的前三个索引 大到小排列
            # 将每个值的第二个值求出 以这些值的概率决定最终第二个的取值是哪个
            for k in range(top3.size):
                d_copy = np.copy(d)  # 拷贝一份d 保证原数据不变
                j = top3[k]  # 取前三大中的一个
                Yg_copy = [j]  # 放入数组
                Z = list(range(self.item_count))
                Z_Y = set(Z).difference(set(Yg_copy))
                # 将其余所有的与最大的那个计算 更新d
                for i in Z_Y:
                    if iter == 0:
                        ei = self.kernel_matrix[j, i] / np.sqrt(d_copy[j])
                    else:
                        ei = (self.kernel_matrix[j, i] - np.dot(c[:iter, j], c[:iter, i])) / np.sqrt(d_copy[j])
                    c[iter, i] = ei  # 更新c数组
                    d_copy[i] = d_copy[i] - ei * ei  # 更新d数组
                d_copy[j] = 0
                beam_container.append(np.argmax(d_copy))  # 将后续要更新的值记录下来 横向计算概率
            # 计算得到的三个值的概率 选概率最大的那个放入Yg中
            beam_container_counter = collections.Counter(beam_container)


if __name__ == "__main__":
    kwargs = {
        'item_count': 100,
        'item_embed_size': 100,
        'max_iter': 100,
        'epsilon': 0.01
    }
    dpp_model = DPPModel(**kwargs)

    #dpp_model.dpp_beam()
    for i in range(8):
        dpp_model.build_kernel_matrix()
        print(dpp_model.dpp())  # 生成八个打散后的视频以及对应概率


