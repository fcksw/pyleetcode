from typing import Optional,List

# k个一组链表反转

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dum = ListNode(next=head)
        prev, end = dum, dum
        while end is not None:
            for i in range(k):
                if end is None:
                    break
                end = end.next
            if end is None:
                break
            start = prev.next
            next_node = end.next
            end.next = None
            prev.next = self.reverse(start)
            start.next = next_node
            end = start
            prev = end
        return dum.next

    def reverse(self, start: Optional[ListNode]) -> Optional[ListNode]:
        prev = ListNode()
        while start is not None:
            next_node = start.next
            start.next = prev
            prev = start
            start = next_node
        return prev

# 两数之和
class Solution:
    def twoSum(self, nums, target: int):
        dic = {}
        for i, num in enumerate(nums):
            if dic.get(target - num) is not None:
                return [i, dic.get(target - num)]
            dic[num] = i

# 盛水最多的容器
# 找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
# 返回容器可以储存的最大水量。
class Solution:
    def maxArea(self, height: List[int]) -> int:
        l, r = 0, len(height) - 1
        maxCap = 0
        while l < r:
            maxCap = max((r - l) * min(height[l], height[r]), maxCap)
            if height[l] < height[r]:
                l += 1
            else:
                r -= 1
        return

# 三数之和
#给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]]
# 满足 i != j、i != k 且 j != k ，同时还满足 nums[i] + nums[j] + nums[k] == 0 。
# 请你返回所有和为 0 且不重复的三元组。
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        for i, num in enumerate(nums):

