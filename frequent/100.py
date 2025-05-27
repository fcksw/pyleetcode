import collections
import copy
from email.header import Header
from itertools import chain
from turtledemo.penrose import start
import time
from typing import List, Dict, Optional


# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:

    #1. 两数之和
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dic = {}
        collections.defaultdict(int)
        for i in range(len(nums)):
            if target - nums[i] in dic:
                return [dic[target - nums[i]], i]
            dic[nums[i]] = i
        return []


    #49. 字母异位词分组
    #输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
    #输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        dic = collections.defaultdict(list)
        for str in strs:
            key = "".join(sorted(str))
            dic[key].append(str)
        return list(dic.values())

    #128. 最长连续序列
    def longestConsecutive(self, nums: List[int]) -> int:
        nums.sort(reverse=False)
        n = len(nums)
        curr, max_num = 0, 0
        for i in range(n):
            if i == 0 or nums[i] - nums[i - 1] == 1:
                curr += 1
            elif nums[i] == nums[i - 1]:
                pass
            else:
                curr = 1
            max_num = max(max_num, curr)
        return max_num


#11. 盛最多水的容器
    def maxArea(self, height: List[int]) -> int:
        n = len(height)
        left, right = 0, n - 1
        max_area = 0
        while left < right:
            max_area = max(max_area, min(height[left], height[right]) * (right - left))
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return max_area

    #15. 三数之和
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        n = len(nums)
        res = []
        for i in range(n - 2):
            """过滤"""
            if i > 0 and nums[i] == nums[i -1]:
                continue
            l, r = i + 1, n - 1
            while l < r:
                sum_num = nums[i] + nums[l] + nums[r]
                if sum_num == 0:
                    res.append([nums[i], nums[l], nums[r]])
                    while l < r and nums[r] == nums[r - 1]:
                        r -= 1
                    while l < r and nums[l] == nums[l + 1]:
                        l += 1
                    l += 1
                    r -= 1
                elif sum_num > 0:
                    r -= 1
                else:
                    l += 1
        return  res

#42. 接雨水
    def trap(self, height: List[int]) -> int:
        n = len(height)
        l, r = 0, n - 1
        l_max, r_max = 0, 0
        res = 0
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

#3. 无重复字符的最长子串
    def lengthOfLongestSubstring(self, s: str) -> int:
        dic: Dict[str, int] = {}
        s_max = 0
        l, r = 0, 0
        while r < len(s):
            if s[r] in dic and dic[s[r]] >= l:
                l = dic[s[r]] + 1
            dic[s[r]] = r
            s_max = max(s_max, r - l + 1)
            r += 1
        return s_max

#438. 找到字符串中所有字母异位词
    def findAnagrams(self, s: str, p: str) -> List[int]:
        p_sort = "".join(sorted(p))
        n = len(p)
        l,r = 0, n - 1
        res = []
        while r < len(s):
            s_cut = "".join(sorted(s[l: r + 1]))
            if s_cut == p_sort:
                res.append(l)
            l += 1
            r += 1
        return res


 #560. 和为 K 的子数组
    def subarraySum(self, nums: List[int], k: int) -> int:
        pre_sum = []
        n = len(nums)
        pre_sum[0] = 0
        for i in range(n):
            """sum[i + 1] = s[i] + nums[i]"""
            pre_sum[i + 1] = pre_sum[i] + nums[i]

        dic = collections.defaultdict(int)
        res = 0
        """ sj - si = k """
        for sj in pre_sum:
            si = sj - k
            if si in dic:
                res += dic[si]
            dic[sj] += 1
        return res


#53. 最大子数组和
    def maxSubArray(self, nums: List[int]) -> int:
        curr, max_sum = nums[0], nums[0]
        for i in range(1, len(nums)):
            curr = max(nums[i], curr + nums[i])
            max_sum = max(curr, max_sum)
        return max_sum

#56. 合并区间
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])
        povit = intervals[0]
        res = []
        for interval in intervals:
            if povit[1] >= interval[0]:
                povit[1] = max(povit[1], interval[1])
            else:
                res.append(povit)
                povit = interval
        res.append(povit)
        return res

#57. 插入区间
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        res = []
        left, right = newInterval[0], newInterval[1]
        merged = False
        for interval in intervals:
            if not merged:
                if right < interval[0]:
                    res.append([left, right])
                    res.append(interval)
                    merged = True
                elif left > interval[1]:
                    res.append(interval)
                else:
                    left = min(left, interval[0])
                    right = max(right, interval[1])
            else:
                res.append(interval)
        if not merged:
            res.append([left, right])
        return res


#189. 轮转数组
    def rotate(self, nums: List[int], k: int) -> None:
        dic = {}
        n = len(nums)
        if k >= n:
            k = k % n
        for i, num in enumerate(nums):
            dic[(i + k) % n] = num

        for key, val in dic.items():
            nums[key] = val

#54. 螺旋矩阵
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        left, right, up, down = 0, len(matrix[0]) - 1, 0, len(matrix) - 1
        res = []
        while True:
            if left > right:
                break
            for i in range(left, right + 1):
                res.append(matrix[i][up])
            up += 1

            if up > down:
                break
            for j in range(up, down + 1):
                res.append(matrix[right][j])
            right -= 1

            if right < left:
                break
            for k in range(right, left - 1, -1):
                res.append(matrix[k][down])
            down -= 1

            if down < up:
                break
            for g in range(down, up - 1, -1):
                res.append(matrix[left][g])
            left += 1
        return res


#48. 旋转图像
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        tmp = copy.deepcopy(matrix)
        n = len(matrix)
        for i in range(n):
            for j in range(n):
                matrix[j][i] = tmp[i][j]


#160. 相交链表
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        A, B = headA, headB
        while A != B:
            A = A.next if A else headB
            B = B.next if B else headA
        return A


#21. 合并两个有序链表
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dum:Optional[ListNode] = ListNode()
        curr = dum
        while list1 and list2:
            v1 = list1.val
            v2 = list2.val
            if v1 <= v2:
                curr.next = list1
                list1 = list1.next
            else:
                curr.next = list2
                list2 = list2.next
            curr = curr.next
        if list1:
            curr.next = list1
        else:
            curr.next = list2
        return dum.next

#206. 反转链表
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        while head:
            next_node = head.next
            head.next = prev
            prev = head
            head = next_node
        return prev

#25. K 个一组翻转链表
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dum = ListNode(val=0, next=head)
        prev, end = dum, dum
        while end:
            for _ in range(k):
                if not end:
                    break
                end = end.next
            if not end:
                break
            start = prev.next
            next_node = end.next
            end.next = None
            prev.next = self.reverseList(start)
            start.next = next_node
            end = start
            prev = start
        return dum.next

#141. 环形链表
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        fast, slow = head, head
        first = True
        while fast and fast.next:
            if fast == slow and not first:
                return True
            first = False
            fast = fast.next.next
            slow = slow.next
        return False


#2. 两数相加
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        res = ListNode()
        dum = res
        tmp = 0
        while l1 or l2:
            l1Val = l1.val if l1 else 0
            l2Val = l2.val if l2 else 0
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
            sum_val = l1Val + l2Val + tmp
            next_val = sum_val % 10
            dum.next = ListNode(val=next_val)
            dum = dum.next
            tmp = sum_val/10
        if tmp > 0:
            dum.next = ListNode(val=tmp)
        return res.next


#19. 删除链表的倒数第 N 个结点
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dum = ListNode(next=head)
        fast, slow = dum, dum
        for _ in range(n):
            if not fast:
                return dum.next
            fast = fast.next
        while fast.next:
            slow = slow.next
            fast = fast.next

        slow.next = slow.next.next
        return dum.next


#24. 两两交换链表中的节点
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dum = ListNode(next=head)
        pre, end = dum, dum
        while end:
            for _ in range(2):
                if not end:
                    break
                end = end.next
            if not end:
                break
            start_node = pre.next
            next_node = end.next
            end.next = start_node
            pre.next = end
            start_node.next = next_node
            pre = start_node
            end = start_node
        return dum.next


#23. 合并 K 个升序链表
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        if len(lists) == 0:
            return None
        elif len(lists) == 1:
            return lists[0]
        povit = lists[0]
        for i in range(1, len(lists)):
            povit = self.mergeSingle(povit, lists[i])
        return povit


    def mergeSingle(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dum = ListNode()
        curr = dum
        while l1 and l2:
            if l1.val <= l2.val:
                curr.next = l1
                l1 = l1.next
            else:
                curr.next = l2
                l2 = l2.next
            curr = curr.next
        if l1:
            curr.next = l1
        else:
            curr.next = l2
        return dum.next

#215. 数组中的第K个最大元素
    def findKthLargest(self, nums: List[int], k: int) -> int:
        if k >= len(nums):
            return -1
        self.findK(0, len(nums) - 1, nums, k)
        return nums[k - 1]


    def findK(self, left: int, right: int, nums: List[int], k):
        p = self.quickSort(left, right, nums)
        if p == k - 1:
            return
        elif p > k - 1:
            self.findK(left, p, nums, k)
        else:
            self.findK(p, right, nums, k)


    def quickSort(self, left: int, right: int, nums: List[int]) -> int:
        pivot = nums[left]
        while left < right:
            while left < right and nums[right] < pivot:
                right -= 1
            if left < right and nums[right] >= pivot:
                nums[left] = nums[right]
                left += 1

            while left < right and nums[left] >= pivot:
                left += 1
            if left < right and nums[left] < pivot:
                nums[right] = nums[left]
                right -= 1
        nums[left] = pivot
        return left

#347. 前 K 个高频元素
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        dic = collections.defaultdict(int)
        max_time = 0
        for num in nums:
            dic[num] += 1
            max_time = max(max_time, dic[num])
        arr = [0] * (max_time + 1)
        for num, time in dic.items():
            arr[time] = num
        res = []
        for i in range(max_time, -1, -1):
            if k == 0:
                break
            if arr[i] == 0:
                continue
            k -= 1
            res.append(arr[i])
        return res











class LinkedNode:
    def __init__(self, key: int, val: int, next: Optional[ListNode], pre: Optional[ListNode]):
        self.key = key
        self.val = val
        self.next = next
        self.pre = pre

class LRUCache:

    def __init__(self, capacity: int):
        self.cache: Dict[int, LinkedNode] = {}
        self.tail = LinkedNode(0, 0, None, None)
        self.head = LinkedNode(0, 0, None, None)
        self.cap = capacity
        self.size = 0
        self.head.next = self.tail
        self.tail.pre = self.head

    def get(self, key: int) -> int:
        if key not in self.cache:
            return  -1
        node = self.cache[key]
        if self.head.next == node:
            return node.val
        self.deleteNode(node)
        self.insertNode(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.val = value
            self.deleteNode(node)
            self.insertNode(node)
        else:
            if self.size >= self.cap:
                self.deleteNode(self.tail.pre)
            node = LinkedNode(key=key, val=value, next=None, pre=None)
            self.insertNode(node)



    def deleteNode(self, node: Optional[LinkedNode]):
        self.size -= 1
        node.pre.next = node.next
        node.next.pre = node.pre
        node.next = None
        node.pre = None
        del self.cache[node.key]

    def insertNode(self, node: Optional[LinkedNode]):
        """ 不再此处作size大小判断 """
        self.size += 1
        self.cache[node.key] = node
        node.next = self.head.next
        node.pre = self.head
        self.head.next.pre = node
        self.head.next = node


#5. 最长回文子串
    def longestPalindrome(self, s: str) -> str:
        n = len(str)
        dp = [[False] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = True
        max_str = s[0]
        for l in range(2, n + 1):
            for i in range(n - l + 1):
                j = i + l - 1
                flag = s[i] == s[j]
                if l == 2:
                    dp[i][j] = flag
                else:
                    dp[i][j] = flag and dp[i + 1][j - 1]
                if dp[i][j]:
                    max_str = s[i : j + 1]
        return max_str

#39. 组合总和
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res, path = [], []
        def dfs(index, remain):
            if remain < 0:
                return
            if remain == 0:
                res.append(path.copy())
            for i in range(index, len(candidates)):
                path.append(candidates[i])
                dfs(i, remain - candidates[i])
                path.pop()
        dfs(0, target)
        return res


#40. 组合总和 II
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        res, path = [], []
        candidates.sort()
        def dfs(index, remain):
            if remain < 0:
                return
            if remain == 0:
                res.append(path.copy())
            for i in range(index, len(candidates)):
                if i > 0 and candidates[i] == candidates[i - 1]:
                    continue
                path.append(candidates[i])
                dfs(i + 1, remain - candidates[i])
                path.pop()
        return res


#113. 路径总和 II
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
            res, path = [], []
            def dfs(node: Optional[TreeNode], remain: int):
                if not node:
                    return
                path.append(node.val)
                if remain - node.val == 0 and not node.left and not node.right:
                    res.append(path.copy())
                    path.pop()
                    return
                dfs(node.left, remain - node.val)
                dfs(node.right, remain - node.val)
                path.pop()
            dfs(root, targetSum)
            return res

    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        self.res = 0
        def dfs(node: Optional[TreeNode]) -> int:
            if not node:
                return 0
            left = dfs(node.left)
            right = dfs(node.right)
            self.res = max(self.res, left + right)
            return max(left, right) + 1
        dfs(root)
        return self.res


    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        left = 0
        n = len(nums)
        res = n + 1
        sum_num = 0
        for right in range(n):
            sum_num += nums[right]
            while sum_num >= target:
                res = min(res, right - left + 1)
                sum_num -= nums[left]
                left += 1
        return res if res < n + 1 else 0






if __name__ == '__main__':
    print(str(int(time.time() * 1000)) )

