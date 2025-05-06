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
        return  res

