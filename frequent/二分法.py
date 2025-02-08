# 34. 在排序数组中查找元素的第一个和最后一个位置
#[5,7,7,8,8,10], 8
from typing import List

from soupsieve.util import lower


class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if not nums:
            return [-1, -1]
        left, right = 0, len(nums) - 1
        first = self.lower(nums, target, left, right)
        if first > right or nums[first] != target:
            return [-1, -1]
        second = self.lower(nums, target + 1, left, right) - 1
        return [first, second]

    def lower(self, nums: List[int], target:int, left: int, right: int):
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] >= target:
                right = mid - 1
            else:
                left = mid + 1
        return left







if __name__ == "__main__":

    test1 = ['a', 'b', 'c']
    test2 = [1,2,3]
    for x in zip(test1, test2):
        print(x)
    pass