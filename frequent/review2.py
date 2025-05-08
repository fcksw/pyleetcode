import collections
from typing import Optional, List, Dict

class ListNode:
    def __init__(self, val, next):
        self.val = val
        self.next = next

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

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


    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dum = ListNode(val=0, next=head)
        slow, fast = dum, dum
        for _ in range(n):
            fast = fast.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next
        slow.next = slow.next.next
        return dum.next

    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dum = ListNode(val=0, next=None)
        curr = dum
        tmp = 0
        while l1 or l2:
            l1_val = l1.val if l1 else 0
            l2_val = l2.val if l2 else 0
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
            sum_num = tmp + l1_val + l2_val
            tmp = sum_num // 10
            curr.next = ListNode(val=sum_num % 10, next=None)
            curr = curr.next
        if tmp > 0:
            curr.next = ListNode(val=tmp, next=None)
        return dum.next


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
            prev.next = self.resver(start)
            start.next = next_node
            end = start
            prev = start
        return dum.next

    def resver(self, head: Optional[ListNode]) -> ListNode:
        prev = None
        while head:
            next_node = head.next
            head.next = prev
            prev = head
            head = next_node
        return prev



    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        def dfs(node: Optional[TreeNode]):
            if not node:
                return
            dfs(node.left)
            res.append(node.val)
            dfs(node.right)
        dfs(root)
        return res

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        def dfs(node: Optional[TreeNode]) -> int:
            if not node:
                return 0
            return max(dfs(node.left), dfs(node.right)) + 1
        return dfs(root)

    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        q = [root]
        res = []
        while q:
            tmp = q
            p, q = [], []
            while tmp:
                node = tmp[0]
                tmp = tmp[1:]
                p.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            res.append(p)
        return res

    def selectK(self, nums: List[int], k):
        self.quickSort(0, len(nums) - 1, k, nums)
        if k - 1 > len(nums) - 1:
            return -1
        return nums[k - 1]

    def quickSort(self, left, right, k, nums:List[int]):
        if left >= right:
            return
        p = self.partition(left, right, nums)
        if p == k - 1:
            return
        elif p > k - 1:
            self.quickSort(left, p - 1, k, nums)
        else:
            self.quickSort(p + 1, right, k, nums)

    def partition(self, left, right, nums:List[int]) -> int:
        povit = nums[left]
        while left < right:
            while left < right and nums[right] >= povit:
                right -= 1
            if left < right and nums[right] < povit:
                nums[left] = nums[right]
                left += 1
            while left < right and nums[left] <= povit:
                left += 1
            if left < right and nums[left] > povit:
                nums[right] = nums[left]
                right -= 1
        nums[left] = povit
        return left



if __name__ == '__main__':
    print((0 + 4 + 6) // 10)
    print((0 + 4 + 6) % 10)
    s = 0 + 4 + 6
    print(s // 10)
    print(s % 10)










