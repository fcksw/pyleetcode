import collections
from typing import List



class Solution:

    # 1. 两数之和
    # 给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出
    # 和为目标值 target  的那 两个 整数，并返回它们的数组下标。
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dic = collections.defaultdict(int)
        for i in range(len(nums)):
            if (target - nums[i]) in dic:
                return [dic[target - nums[i]], i]
            dic[nums[i]] = i
        return []




    # 128. 最长连续序列
    # 给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。
    def longestConsecutive(self, nums: List[int]) -> int:
        if not nums:
            return 0
        nums.sort()
        max_len, curr = 1, 1
        for i in range(1, len(nums)):
            if nums[i] - nums[i - 1] == 1:
                curr += 1
            elif nums[i] - nums[i - 1] == 0:
                continue
            else:
                curr = 1
            max_len = max(max_len, curr)
        return max_len





    # 49. 字母异位词分组
    # 给你一个字符串数组，请你将 字母异位词 组合在一起。可以按任意顺序返回结果列表。
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        dic = collections.defaultdict(List)
        for str in strs:
            key = "".join(sorted(str))
            dic[key].append(str)
        res = []
        for v in dic.values():
            res.append(v)
        return res


    # 283. 移动零
    # 输入: nums = [0,1,0,3,12]
    # 输出: [1,3,12,0,0]
    def moveZeroes(self, nums: List[int]) -> None:
        left = 0
        for right in range(len(nums)):
            if nums[right] != 0:
                tmp = nums[right]
                nums[right] = 0
                nums[left] = tmp
                left += 1


    # 209. 长度最小的子数组
    # 给定一个含有 n 个正整数的数组和一个正整数 target 。
    # 找出该数组中满足其总和大于等于 target 的长度最小的
    # 子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。
    # 如果不存在符合条件的子数组，返回 0 。
    def smallestArray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        left = 0
        min_len = n + 1
        curr = 0
        for right in range(n):
            curr += nums[right]
            while left <= right and curr >= target:
                min_len = min(min_len, right - left + 1)
                curr -= nums[left]
                left += 1
        return min_len




