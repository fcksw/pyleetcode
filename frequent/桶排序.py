import collections
from typing import List

# 347. 前 K 个高频元素
def topKFrequent(self, nums: List[int], k: int) -> List[int]:
    m, max_num = len(nums), 0
    """ 记录单词出现次数 a:3 b:4 c:5 """
    dic = collections.defaultdict(int)
    for i in range(m):
        dic[nums[i]] += 1
        max_num = max(max_num, dic[nums[i]])
    tmp = [[] for _ in range(max_num + 1)]
    for key, value in dic.items():
        tmp[value].append(key)
    res = []
    for i in range(max_num, -1, -1):
        tp = tmp[i]
        if not tp:
            continue
        if k == 0:
            break
        for j in range(len(tp)):
            k -= 1
            res.append(tp[j])
    return res


