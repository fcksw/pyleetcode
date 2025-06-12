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


    # 209 长度最小的子数组
    """
        给定一个含有 n 个正整数的数组和一个正整数 target 。
        找出该数组中满足其总和大于等于 target 的长度最小的 子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，
        并返回其长度。如果不存在符合条件的子数组，返回 0 。
        输入：target = 7, nums = [2,3,1,2,4,3]
        输出：2
        解释：子数组 [4,3] 是该条件下的长度最小的子数组。
    """
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

    def lengthOfLongestSubstring(self, s: str) -> int:
        left, res = 0, 0
        n = len(s)
        dic = collections.defaultdict(int)
        for right in range(n):
            if s[right] in dic and dic[s[right]] >= left:
                left = dic[s[right]] + 1
            res = right - left + 1
            dic[s[right]] = right
        return res