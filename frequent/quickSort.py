# 快排
# 快排文档 https://leetcode.cn/problems/kth-largest-element-in-an-array/description/

from typing import List


class Solution:

    def quickSort(self, left: int, right: int, nums: List[int]):
        if left < right:
            pivot = self.partition(left, right, nums)
            self.quickSort(left, pivot - 1, nums)
            self.quickSort(pivot + 1, right, nums)


    def partition(self, left: int, right: int, nums: List[int]) -> int:
        pivot = nums[left]
        while left < right:
            while nums[right] >= pivot and left < right:
                right -= 1
            if nums[right] < pivot and left < right:
                nums[left] = nums[right]
                left += 1

            while nums[left] <= pivot and left < right:
                left += 1
            if nums[left] > pivot and left < right:
                nums[right] = nums[left]
                right -= 1
        nums[left] = pivot
        return left



    def findKthLargest(self, nums: List[int], k: int) -> int:
        self.quickSort(0, len(nums) - 1, k, nums)
        return nums[k - 1]

    def quickSort(self, left: int, right: int, k: int, nums: List[int]):
        res = self.partitionKth(left, right, nums)
        if res == k - 1:
            return
        elif res < k - 1:
            self.quickSort(res + 1, right, k, nums)
        else:
            self.quickSort(left, res - 1, k, nums)



    def partitionKth(self, left:int, right:int, nums:List[int]) -> int:
        pivot = nums[left]
        while left < right:
            while left < right and nums[right] <= pivot:
                right -= 1
            if left < right and nums[right] > pivot:
                nums[left] = nums[right]
                left += 1

            while left < right and nums[left] >= pivot:
                left += 1
            if left < right and nums[left] < pivot:
                nums[right] = nums[left]
                right -= 1
        nums[left] = pivot
        return left



if __name__ == "__main__":
    name = "abc"
    print(f'llal {name}')
    print((-1)/2)
    # listSort = [1,4,6,32,5,0,3,12,53,223]
    # s = Solution()
    # s.quickSort(0, len(listSort) - 1, listSort)
    # print(listSort)
