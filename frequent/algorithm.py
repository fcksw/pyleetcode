from typing import Optional,List
import copy

from jedi.plugins.django import mapping


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
    def twoSum(self, nums, target: int):
        dic = {}
        for i, num in enumerate(nums):
            if dic.get(target - num) is not None:
                return [i, dic.get(target - num)]
            dic[num] = i

# 盛水最多的容器
# 找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
# 返回容器可以储存的最大水量。
    def maxArea(self, height: List[int]) -> int:
        l, r = 0, len(height) - 1
        max_cap = 0
        while l < r:
            max_cap = max((r - l) * min(height[l], height[r]), max_cap)
            if height[l] < height[r]:
                l += 1
            else:
                r -= 1
        return max_cap

# 三数之和
#给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]]
# 满足 i != j、i != k 且 j != k ，同时还满足 nums[i] + nums[j] + nums[k] == 0 。
# 请你返回所有和为 0 且不重复的三元组。
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        # TODO 优先排序
        nums = sorted(nums)
        res = []
        i, max_len = 0, len(nums) - 1
        while i <= max_len - 2:
            if i > 0 and nums[i] == nums[i - 1]:
                i += 1
                continue
            l, r = i + 1, len(nums) - 1
            while l < r:
                sum_num = nums[i] + nums[l] + nums[r]
                if sum_num > 0:
                    r -= 1
                elif sum_num < 0:
                    l += 1
                else:
                    res.append([nums[i], nums[l], nums[r]])
                    while l < r and l < max_len and nums[l] == nums[l + 1]:
                        l += 1
                    while r > l and r > i and nums[r] == nums[r - 1]:
                        r -= 1
                    r -= 1
                    l += 1
            i += 1
        return res

# 接雨水
    def trap(self, height: List[int]) -> int:
        ans, l, r  = 0, 0, len(height) - 1
        l_max, r_max = 0, 0
        while l < r:
            l_max = max(height[l], l_max)
            r_max = max(height[r], r_max)
            if height[l] < height[r]:
                ans += l_max - height[l]
                l += 1
            else:
                ans += r_max - height[r]
                r -= 1
        return ans

# 3. 无重复字符的最长子串
    def lengthOfLongestSubstring(self, s: str) -> int:
        dict_s = {}
        l, r, max_len = 0, 0, 0
        while r < len(s):
            if s[r] in dict_s and dict_s[s[r]] >= l:
                l = dict_s[s[r]] + 1
            max_len = max(r - l + 1, max_len)
            dict_s[s[r]] = r
            r += 1

        return max_len

# 53. 最大子数组和
    def maxSubArray(self, nums: List[int]) -> int:
        max_num, pre = nums[0], nums[0]
        i = 1
        while i < len(nums):
            pre = max(nums[i], pre + nums[i])
            max_num = max(pre, max_num)
            i += 1
        return max_num

# 56. 合并区间
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])
        if len(intervals) == 1:
            return intervals
        base_interval, ans = intervals[0], []
        i = 1
        while i < len(intervals):
            interval = intervals[i]
            if base_interval[1] < interval[0]:
                ans.append(base_interval)
                base_interval = interval
            else:
                base_interval[1] = max(base_interval[1], interval[1])
            i += 1
        ans.append(base_interval)
        return ans


# 189. 轮转数组
#     输入: nums = [1, 2, 3, 4, 5, 6, 7], k = 3
#     输出: [5, 6, 7, 1, 2, 3, 4]
    def rotate(self, nums: List[int], k: int) -> None:
        for _ in range(k):
            num = nums.pop()
            nums.insert(0, num)

    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        # (0,0) (0, 1) (0,2)
        # (2,0) (1,0) (0,0)
        tmp = copy.deepcopy(matrix)
        for i in range(len(matrix[0])):
            for j in range(len(matrix)):
                matrix[i][j] = tmp[len(matrix) - 1 - j][i]


# 54. 螺旋矩阵
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        """  """
        left, right, down, up = 0, len(matrix[0]) - 1, len(matrix) - 1, 0
        ans = []
        while True:
            if left > right:
                break
            for i in range(left, right + 1):
                ans.append(matrix[up][i])
            up += 1

            if up > down:
                break
            for i in range(up, down + 1):
                ans.append(matrix[i][right])
            right -= 1

            if left > right:
                break
            for i in range(right, left - 1, -1):
                ans.append(matrix[down][i])
            down -= 1

            if up > down:
                break
            for i in range(down, up - 1, -1):
                ans.append(matrix[i][left])
            left += 1
        return ans

# 160. 相交链表
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        dic = {}
        while headA is not None:
            dic[headA] = True
            headA = headA.next

        while headB is not None:
            if headB in dic:
                return headB
            headB = headB.next
        return None

# 206. 反转链表
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        while head is not None:
            next_node = head.next
            head.next = prev
            prev = head
            head = next_node
        return prev

# 141. 环形链表
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        fast, slow = head, head
        first = True
        while fast is not None and fast.next is not None:
            if slow == fast and not first:
                return True
            first = False
            slow = slow.next
            fast = fast.next.next
        return False

# 21. 合并两个有序链表
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dum = ListNode()
        res = dum
        while list1 is not None and list2 is not None:
            l1_val = list1.val
            l2_val = list2.val
            if l1_val > l2_val:
                dum.next = ListNode(val=l2_val)
                list2 = list2.next
            else:
                dum.next = ListNode(val=l1_val)
                list1 = list1.next
            dum = dum.next

        if list1 is None:
            dum.next = list2
        else:
            dum.next = list1
        return res.next

# 2. 两数相加
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        tmp, pre = 0, ListNode()
        res = pre
        while l1 is not None or l2 is not None:
            l1_val, l2_val = 0, 0
            if l1 is not None:
                l1_val = l1.val
                l1 = l1.next
            if l2 is not None:
                l2_val = l2.val
                l2 = l2.next
            val = l1_val + l2_val + tmp
            tmp = int(val / 10)
            pre.next = ListNode(val= val % 10)
            pre = pre.next

        if tmp != 0:
            pre.next = ListNode(val = tmp)
        return  res.next

# 19. 删除链表的倒数第 N 个结点
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dum = ListNode(0, head)
        fast, slow = dum, dum
        for _ in range(n):
            fast = fast.next

        while fast.next is not None:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return dum.next

# 24. 两两交换链表中的节点
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dum = ListNode(val=0, next=head)
        pre, end = dum, dum
        while end is not None:
            for _ in range(2):
                if end is None:
                    return dum.next
                end = end.next

            if end is None:
                return dum.next
            next_node = end.next
            start = pre.next
            start.next = next_node
            end.next = start
            pre.next = end
            pre, end = start, start
        return dum.next

# k个一组链表反转
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dum = ListNode(val=0, next=head)
        pre, end = dum, dum
        while end is not None:
            for _ in range(k):
                if end is None:
                    return dum.next
                end = end.next
            if end is None:
                return dum.next
            next_node = end.next
            end.next = None
            start = pre.next
            pre.next = self.reverse(start)
            start.next = next_node
            pre, end = start, start

        return dum.next

    def reverse(self, start: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        while start is not None:
            next_node = start.next
            start.next = prev
            prev = start
            start = next_node
        return prev




if __name__ == '__main__':
    print('Hello World')


