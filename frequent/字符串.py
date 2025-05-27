import collections
from typing import List


class Solution:
    # 395. 至少有 K 个重复字符的最长子串
    def longestSubstring(self, s: str, k: int) -> int:
        n = len(s)
        res = 0
        for i in range(n):
            dic = collections.defaultdict(int)
            num = 0
            for j in range(i, n):
                dic[s[j]] += 1
                if dic[s[j]] == k:
                    num += 1
                if num == len(dic):
                    res = max(res, j - i + 1)
        return res


    # 209
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        left = 0
        n = len(nums)
        res = n + 1
        sum_num = 0
        for right in range(n):
            sum_num += nums[right]
            while sum_num >= target:
                res = min(res, right - left + 1)
                sum_num -= nums[left]
                left += 1
        return res if res < n + 1 else 0

