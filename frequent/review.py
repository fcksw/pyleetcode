from typing import Optional, List
from math import inf


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    # k个一组链表反转
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dum = ListNode(next=head)
        prev, end = dum, dum
        while end:
            for i in range(k):
                if end:
                    end = end.next
            if not end:
                return dum.next

            next_node = end.next
            end.next = None
            start_node = prev.next
            # 重新连接
            prev.next = self.reverse(start_node)
            start_node.next = next_node
            end = start_node
            prev = start_node
        return dum.next

    def reverse(self, start: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        while start:
            next_node = start.next
            start.next = prev
            prev = start
            start = next_node
        return prev

    def twoSum(self, nums, target: int):
        dic = {}
        for i, item in enumerate(nums):
            if target - item in dict:
                return [i, dict[target - item]]
            dic[item] = i

    # 盛水最多的容器
    # 找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
    # 返回容器可以储存的最大水量。
    def maxArea(self, height: List[int]) -> int:
        max_cap = 0
        l, r = 0, len(height) - 1
        while l < r:
            max_cap = max((r - l) * min(height[l], height[r]), max_cap)
            if height[l] < height[r]:
                l += 1
            else:
                r -= 1
        return max_cap

    # 三数之和
    # 给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]]
    # 满足 i != j、i != k 且 j != k ，同时还满足 nums[i] + nums[j] + nums[k] == 0 。
    # 请你返回所有和为 0 且不重复的三元组。
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []
        i, max_len = 0, len(nums) - 1
        while i <= max_len - 2:
            # i去重
            if i > 0 and nums[i] == nums[i - 1]:
                i += 1
                continue
            left, right = i + 1, len(nums) - 1
            while left < right:
                # left, right去重
                sum_three = nums[i] + nums[left] + nums[right]
                if sum_three == 0:
                    res.append([nums[i], nums[left], nums[right]])
                    while left < max_len and nums[left] == nums[left + 1]:
                        left += 1
                    while right > i and nums[right] == nums[right -1]:
                        right -= 1
                    left += 1
                    right -= 1
                elif sum_three > 0:
                    right -= 1
                else:
                    left += 1
            i += 1
        return res


    # 最长公共前缀
    def longestCommonPrefix(self, strs: List[str]) -> str:
        stan = strs[0]
        for i in range(1, len(strs)):
            stan = self.twoCommonPrefix(stan, strs[i])
            if not stan:
                return stan
        return stan

    def twoCommonPrefix(self, str1: str, str2: str) -> str:
        res = ""
        for i in range(min(len(str1), len(str2))):
            if str1[i] != str2[i]:
                break
            else:
                res = str1[0:i + 1]
        return res


    # 接雨水
    def trap(self, height: List[int]) -> int:
        l, r = 0, len(height) - 1
        res = 0
        l_max, r_max = 0, 0
        while l < r:
            l_max = max(l_max, height[l])
            r_max = max(r_max, height[r])
            if height[l] < height[r]:
                res += l_max - height[l]
                l += 1
            else:
                res += r_max - height[r]
                r -= 1
        return res

    # 3. 无重复字符的最长子串
    def lengthOfLongestSubstring(self, s: str) -> int:
        dic = {}
        l = 0
        max_len = 0
        for r in range(len(s)):
            if s[r] in dic and dic[s[r]] >= l:
                l = dic[s[r]] + 1
            max_len = max(r - l + 1, max_len)
            dic[s[r]] = r
        return max_len

    # 53. 最大子数组和
    def maxSubArray(self, nums: List[int]) -> int:
        curr, max_num = nums[0], nums[0]
        for i in range(1, len(nums)):
            curr = max(nums[i], curr + nums[i])
            max_num = max(curr, max_num)
        return max_num

if __name__ == "__main__":
    s = Solution()
    s.longestCommonPrefix(["flower","flow","flight"])