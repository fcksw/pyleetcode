from typing import List


class Solution:


    # 718. 最长重复子数组
    #  i , j 为 nums1 nums2 的最后一个节点
    #  nums1 从0 到 i 与 nums2从 0 到 j 的最长子数组
    # 前提 nums1[i] == nums2[j]
    #  dp[i][j] = max()
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        res = 0
        m, n = len(nums1), len(nums2)
        dp = [[0] * m for _ in range(n)]
        for i in range(m):
            """ 0,1 0,2 """
            if nums2[0] == nums1[i]:
                dp[0][i] = 1
                res = 1
        for i in range(n):
            """ 0,1 0,2 """
            if nums1[0] == nums2[i]:
                dp[i][0] = 1
                res = 1
        for i in range(1, n):
            for j in range(1, m):
                if nums1[j] == nums2[i]:
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - 1] + 1)
                    res = max(res, dp[i][j])
        return res

    # 5. 最长回文子串
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = True
        max_str = s[0]
        for l in range(2, n + 1):
            for i in range(n):
                j = i + l - 1
                if j >= n:
                    break
                if l == 2:
                    dp[i][j] = s[i] == s[j]
                else:
                    dp[i][j] = (s[i] == s[j]) and dp[i + 1][j - 1]
                if dp[i][j]:
                    max_str = s[i:j + 1]
        return max_str


    # 300. 最长递增子序列
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [1] * n
        res = 1
        for i in range(1, n):
            for j in range(0, i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
                    res = max(res, dp[i])
        return res

    """给你两个下标从 0 开始的数组 present 和 future ，present[i] 和 future[i] 
    分别代表第 i 支股票现在和将来的价格。每支股票你最多购买 一次 ，你的预算为 budget
    求最大的收益。
    """
    # 0-1背包问题
    # 2291. 最大股票收益
    def maximumProfit(self, present: List[int], future: List[int], budget: int) -> int:
        m, n = len(present), budget + 1 #金额
        dp = [[0] * n for _ in range(m)]
        for v in range(n + 1):
            if v >= present[0]:
                dp[0][v] = max(0, future[0] - present[0])
        for i in range(m):
            for j in range(n + 1):
                if j < present[i]:
                    dp[i][j] = dp[i - 1][j]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - present[i]] + future[i] - present[i])
        return dp[m - 1][n]
