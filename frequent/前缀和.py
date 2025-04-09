##https://leetcode.cn/problems/range-sum-query-immutable/solutions/2693498/qian-zhui-he-ji-qi-kuo-zhan-fu-ti-dan-py-vaar/
##
from typing import List, Dict
import collections

# 例题：560. 和为 K 的子数组
def subarraySum(self, nums: List[int], k: int) -> int:
    n = len(nums)
    pre_sum = [0] * (n + 1)
    pre_sum[0] = 0
    for i in range(n):
        """ s[i + 1] = s[i] + nums[i] """
        pre_sum[i + 1] = pre_sum[i] + nums[i]
    dic: Dict[int, int] = collections.defaultdict(int)
    res = 0
    for sj in pre_sum:
        si = sj - k
        if si in dic:
            res += dic[si]
        """ 如果 pre_num[n] = sj, pre_num[m] = sj 的值相同，那么dic[sj] 就会出现两次，
            res += dic[si]
            和
            dic[sj] += 1
            
            即 在sj 之前，si出现的次数
         """
        dic[sj] += 1
    return res

