from typing import Optional,List, Dict
import copy
from math import inf

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

# 23. 合并 K 个升序链表
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        if lists is None or len(lists) <= 1:
            return lists
        origin = lists[0]
        for item in lists[1:]:
            origin = self.mergeList(origin, item)
        return origin

    def mergeList(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        res = ListNode()
        merge_list = res
        while l1 is not None and l2 is not None:
            l1_val = l1.val
            l2_val = l2.val
            if l1_val < l2_val:
                merge_list.next = l1
                l1 = l1.next
            else:
                merge_list.next = l2
                l2 = l2.next
            merge_list = merge_list.next
        if l1 is None:
            merge_list.next = l2
        else:
            merge_list.next = l1
        return  res.next

# 25. K 个一组翻转链表
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dum = ListNode(next=head)
        pre, end = dum, dum
        while head is not None:
            for _ in range(k):
                if end is None:
                    return dum.next
                end = end.next
            if end is None:
                return dum.next
            next_node = end.next
            end.next = None
            start_node = pre.next
            pre.next = self.reverseSingle(start_node)
            start_node.next = next_node
            pre, end = start_node, start_node
        return dum.next


    def reverseSingle(self, prev: Optional[ListNode]) -> Optional[ListNode]:
        pre = None
        if prev is not None:
            next_node = prev.next
            prev.next = pre
            pre = prev
            prev = next_node
        return pre



class LinkedNode:
    def __init__(self, key, value, pre, next):
        self.key = key
        self.value = value
        self.pre = pre
        self.next = next


# 146. LRU 缓存
class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.mapCache: Dict[int, LinkedNode] = {}
        self.head = LinkedNode(0, 0, None, None)
        self.tail = LinkedNode(0, 0, None, None)
        self.head.next = self.tail
        self.tail.pre = self.head

    def get(self, key: int) -> int:
        if key not in self.mapCache:
            return -1
        # 将数据移动到最前端，然后返回
        node = self.mapCache[key]
        self.__del_node(node)
        self.__insert_node_to_head(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        # todo judge cap and size
        if key in self.mapCache:
            exist_node = self.mapCache[key]
            self.__del_node(exist_node)
            self.__insert_node_to_head(exist_node)
            exist_node.value = value
            return
        if self.size == self.capacity:
            self.__del_node(self.tail.pre)

        node = LinkedNode(key, value, None, None)
        self.__insert_node_to_head(node)


    def __del_node(self, node: Optional[LinkedNode]):
        pre_node = node.pre
        next_node = node.next
        pre_node.next = next_node
        next_node.pre = pre_node
        node.next = None
        node.pre = None
        del self.mapCache[node.key]
        self.size -= 1

    def __insert_node_to_head(self, node: Optional[LinkedNode]):
        next_node = self.head.next
        self.head.next = node
        node.pre = self.head
        next_node.pre = node
        node.next = next_node
        self.size += 1
        self.mapCache[node.key] = node


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class TreeSolution:

# 94. 二叉树的中序遍历
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        if not root:
            return res
        def dfs(node: Optional[TreeNode]):
            if not node:
                return
            dfs(node.left)
            res.append(node.val)
            dfs(node.right)
        dfs(root)
        return res


# 104. 二叉树的最大深度
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        def dfs(node) -> int:
            if not node:
                return 0
            return max(dfs(node.left), dfs(node.right)) + 1
        return dfs(root)


# 102. 二叉树的层序遍历
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        res = [[]]
        if not root:
            return res
        q = [root]
        while q:
            tmp = q
            q = []
            dum = []
            while tmp:
                node = tmp[0]
                dum.append(node.val)
                tmp = tmp[1:]
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            res.append(dum)
        return res


# 98. 验证二叉搜索树
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def dfs(node: Optional[TreeNode], max_val, min_val) -> bool:
            if not node:
                return True
            if node.val <= min_val or node.val >= max_val:
                return False
            return dfs(node.left, node.val, min_val) and dfs(node.right, max_val, node.val)
        return dfs(root, inf, -inf)


# 230. 二叉搜索树中第 K 小的元素
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        res: int
        def dfs(node: Optional[TreeNode]):
            if not node:
                return
            dfs(node.left)
            if self.k == 1:
                self.res = node.val
            self.k -= 1
            dfs(node.right)
        self.k = k
        dfs(root)
        return self.res

# 114. 二叉树展开为链表
    def flatten(self, root: Optional[TreeNode]) -> None:
        self.res = []
        def dfs(node: Optional[TreeNode]):
            if not node:
                return
            self.res.append(node)
            dfs(node.left)
            dfs(node.right)
        dfs(root)
        for i in range(0, len(self.res) - 1):
            pre_node = self.res[i]
            pre_node.left = None
            pre_node.right = self.res[i + 1]
        return self.res
# 105. 从前序与中序遍历序列构造二叉树
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:


if __name__ == '__main__':
    for i in range(0, 10, 2):
        print(i)


