from typing import List


def maxChild(nums:List[int]) -> int:
    n = len(nums)
    dp = [1] * n
    res = 0
    for i in range(1, n):
        for j in range(0, i):
            if nums[i] > nums[j]:
                dp[i] = dp[j] + 1
                res = max(dp[i], res)
    return res


print(maxChild([1,3,5,2,5]))
