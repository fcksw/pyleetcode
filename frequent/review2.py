import collections
from typing import Optional, List, Dict

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class LinkedNode:
    def __init__(self, key:int, value:int, next_node, pre_node):
        self.key = key
        self.value = value
        self.next_node = next_node
        self.pre_node = pre_node


class LRUCache:

    def __init__(self, capacity: int):
        self.cap = capacity
        self.size = 0
        self.cache = {}
        self.head = LinkedNode(0,0,None,None)
        self.tail = LinkedNode(0,0,None,None)
        self.head.next_node = self.tail
        self.tail.pre_node = self.head

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        """删除node size-- cache del 插入node size++ cache insert"""
        self._del_node(node)
        self._insert_node(node)
        return node.value


    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self._del_node(node)
            self._insert_node(node)
        else:
            node = LinkedNode(key,value, None, None)
            self._insert_node(node)


    def _del_node(self, node: LinkedNode):
        self.size -= 1
        del self.cache[node.key]
        node.next_node.pre_node = node.pre_node
        node.pre_node.next_node = node.next_node
        node.next_node = None
        node.pre_node = None

    def _insert_node(self, node: LinkedNode):
        if self.size >= self.cap:
            self._del_node(self.tail.pre_node)
        self.size += 1
        self.cache[node.key] = node
        node.next_node = self.head.next_node
        node.pre_node = self.head
        self.head.next_node = node
        node.next_node.pre_node = node

class Solution:

    def subarraySum(self, nums: List[int], k: int) -> int:
        pre_sum = [0 * len(nums) + 1]
        for i in range(len(nums)):
            pre_sum[i + 1] = pre_sum[i] + nums[i]
        dic: Dict[int, int] = collections.defaultdict(int)
        res = 0
        for sj in pre_sum:
            si = sj - k
            if si in dic:
                res += dic[si]
            dic[sj] += 1
        return res

    # 34. 在排序数组中查找元素的第一个和最后一个位置
    # [5,7,7,8,8,10], 8
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if not nums:
            return []
        left = self.part(0, len(nums) - 1, target, nums)
        if left < 0 or left >= len(nums) or nums[left] != target:
            return [-1, -1]
        right = self.part(0, len(nums) - 1, target + 1, nums) - 1
        return [left, right]

    def part(self, left, right, target, nums:List[int]) -> int:
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] >= target:
                right = mid - 1
            else:
                left = mid + 1
        return left

    def trap(self, height: List[int]) -> int:
        n = len(height)
        l, r = 0, n - 1
        l_max, r_max = 0, 0
        res = 0
        while l < r:
            l_max = max(l_max, height[l])
            r_max = max(r_max, height[r])
            if height[l] > height[r]:
                res += r_max - height[r]
                r -= 1
            else:
                res += l_max - height[l]
                l += 1
        return res

    def longestConsecutive(self, nums: List[int]) -> int:
        nums.sort()
        curr, max_len = 1, 1
        for index in range(1, len(nums)):
            if nums[index] - nums[index - 1] == 1:
                curr += 1
            elif nums[index] - nums[index - 1] == 0:
                continue
            else:
                curr = 1
            max_len = max(max_len, curr)
        return max_len


    def subarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        pre_sum = [0 * (n + 1)]
        for i in range(n):
            pre_sum[i + 1] = pre_sum[i] + nums[i]

        res = 0
        dic = collections.defaultdict(int)
        for i in range(n + 1):
            si = pre_sum[i] - k
            if si in dic:
                res += dic[si]
            dic[pre_sum[i]] += 1
        return res

    def lengthOfLongestSubstring(self, s: str) -> int:
        dic = collections.defaultdict(int)
        l, res = 0, 0
        for r in range(len(s)):
            if s[r] in dic and dic[s[r]] >= l:
                l = dic[s[r]] + 1
            dic[s[r]] = r
            res = max(res, r - l + 1)
        return res

    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        A, B = headA, headB
        while A != B:
            A = A.next if A else headB
            B = B.next if B else headA
        return A

