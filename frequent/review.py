import collections
import os
from collections import defaultdict
from typing import Optional,List,Dict
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

    # 56. 合并区间
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals = sorted(intervals, key = lambda x : x[0])
        start = intervals[0]
        res = []
        for index in range(1, len(intervals)):
            interval = intervals[index]
            if start[1] < interval[0]:
                res.append(start)
                start = interval
            else:
                start[1] = max(start[1], interval[1])
        res.append(start)
        return res

    # 189. 轮转数组
        #     输入: nums = [1, 2, 3, 4, 5, 6, 7], k = 3
        #     输出: [5, 6, 7, 1, 2, 3, 4]
    def rotate(self, nums: List[int], k: int) -> None:
        dic = {}
        if k > len(nums):
            k = k % len(nums)
        for i, num in enumerate(nums):
            if i >= len(nums) - k:
                dic[i - (len(nums) - k)] = num
            else:
                dic[i + k] = num
        for key, val in dic.items():
            nums[key] = val

    # 54. 螺旋矩阵
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        up, down, l, r = 0, len(matrix) - 1, 0, len(matrix[0]) - 1
        res = []
        while True:
            if l > r:
                break
            for i in range(l, r + 1):
                res.append(matrix[up][i])
            up += 1

            if up > down:
                break
            for i in range(up, down + 1):
                res.append(matrix[i][r])
            r -= 1

            if r < l:
                break
            for i in range(r, l - 1, -1):
                res.append(matrix[down][i])
            down -= 1

            if down < up:
                break
            for i in range(down, up - 1, -1):
                res.append(matrix[i][l])
            l += 1
        return res

    # 160. 相交链表
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        dic = {}
        while headA:
            dic[headA] = 0
            headA = headA.next
        while headB:
            if headB in dic:
                return headB
            headB = headB.next
        return None

    # 206. 反转链表
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        while head:
            next_node = head.next
            head.next = prev
            prev = head
            head = next_node
        return prev

# 141. 环形链表
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow, fast = head, head
        while fast and fast.next:
            if fast is slow:
                return True
            slow = slow.next
            fast = fast.next.next
        return False

# 21. 合并两个有序链表
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dum = ListNode()
        prev = dum
        while list1 and list2:
            if list1.val <= list2.val:
                prev.next = list1
                list1 = list1.next
            else:
                prev.next = list2
                list2 = list2.next
            prev = prev.next
        if list1:
            prev.next = list1
        else:
            prev.next = list2
        return dum.next


# 2. 两数相加
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        tmp = 0
        dum = ListNode()
        res = dum
        while l1 or l2:
            l1_val, l2_val = 0, 0
            if l1:
                l1_val = l1.val
                l1 = l1.next
            if l2:
                l2_val = l2.val
                l2 = l2.next
            total = l1_val + l2_val + tmp
            res.next = ListNode(total % 10)
            tmp = total//10
            res = res.next
        if tmp > 0:
            res.next = ListNode(tmp)
        return dum.next

# 19. 删除链表的倒数第 N 个结点
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dum = ListNode(val=0, next=head)
        prev, end = dum, dum
        for _ in range(0, n + 1):
            if not end:
                return []
            end = end.next
        while end and not end.next:
            prev = prev.next
            end = end.next
        prev.next = None
        return dum.next


# 24. 两两交换链表中的节点
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dum = ListNode(val=0, next=head)
        prev, end = dum, dum
        while end:
            for _ in range(2):
                if not end:
                    return dum.next
                end = end.next
            if not end:
                return dum.next
            next_node = end.next
            end.next = None
            start_node = prev.next
            prev.next = end
            end.next = start_node
            start_node.next = next_node
            prev, end = start_node, start_node
        return dum.next


# k个一组链表反转
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dum = ListNode(next=head)
        prev, end = dum, dum

        while end:
            for _ in range(k):
                if not end:
                    return dum.next
                end = end.next
            if not end:
                return dum.next

            next_node = end.next
            end.next = None
            start_node = prev.next
            prev.next = self.reverseSingle(start_node)
            start_node.next = next_node
            prev, end = start_node, start_node
        return dum.next

    def reverseSingle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        while head:
            next_node = head.next
            head.next = prev
            prev = head
            head = next_node
        return prev

class LinkedNode:
    def __init__(self, key, value, pre, next):
        self.key: int = key
        self.value = value
        self.pre = pre
        self.next = next

# 146. LRU 缓存
class LRUCache:

    def __init__(self, capacity: int):
        self.size = 0
        self.capacity = capacity
        self.cache: Dict[int, LinkedNode] = {}
        self.head = LinkedNode(0, 0, None, None)
        self.tail = LinkedNode(0, 0, None, None)
        self.head.next = self.tail
        self.tail.pre = self.head


    def get(self, key: int) -> int:
        """ 1、移动到最前端  2、返回数据 """
        if key not in self.cache:
            return -1
        """ 1、删除节点 2、插入节点 """
        node = self.cache[key]
        self.__del_node(node)
        self.__insert_node(node)
        return node.value


    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            self.__del_node(node)
            self.__insert_node(node)
            node.value = value
        else:
            node = LinkedNode(key, value, None, None)
            self.__insert_node(node)



    def __del_node(self, node: LinkedNode):
        """size-- del cache"""
        node.pre.next = node.next
        node.next.pre = node.pre
        node.next = None
        node.pre = None
        del self.cache[node.key]
        self.size -= 1


    def __insert_node(self, node: LinkedNode):
        """ 如果超过cap，删除最不常用的 """
        if self.size == self.capacity:
            self.__del_node(self.tail.pre)
        node.next = self.head.next
        node.pre = self.head
        self.head.next = node
        node.next.pre = node
        self.size += 1
        self.cache[node.key] = node

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
            return []
        def dsf(node: Optional[TreeNode]):
            if not node:
                return
            dsf(node.left)
            res.append(node.val)
            dsf(node.right)
        dsf(root)
        return res

# 104. 二叉树的最大深度
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        def depth(node: Optional[TreeNode]):
            if not node:
                return 0
            return max(depth(node.left), depth(node.right)) + 1
        return depth(root)


# 102. 二叉树的层序遍历
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        res = []
        q: List[Optional[TreeNode]] = []
        q.append(root)
        while q:
            tmp = q
            q = []
            tmp_res = []
            while tmp:
                node = tmp[0]
                tmp_res.append(node.val)
                tmp = tmp[1:]
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            res.append(tmp_res)
        return res

# 98. 验证二叉搜索树
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def dfs(node: Optional[TreeNode], max_val, min_val) -> bool:
            if not node:
                return True
            if node.val > max_val or node.val < min_val:
                return False
            return dfs(node.left, node.val, min_val) and dfs(node.right, max_val, node.val)
        return dfs(root, inf, -inf)

# 230. 二叉搜索树中第 K 小的元素
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        self.k = k
        def dfs(node:Optional[TreeNode]):
            if not node:
                return
            dfs(node.left)
            if self.k == 1:
                self.res = node.val
            self.k -= 1
            dfs(node.right)
        return self.res

# 114. 二叉树展开为链表
    def flatten(self, root: Optional[TreeNode]) -> None:
        self.res: List[Optional[TreeNode]] = []
        def dfs(node: Optional[TreeNode]):
            if not node:
                return
            self.res.append(node)
            dfs(node.left)
            dfs(node.right)
        dfs(root)
        for i in range(len(self.res) - 1):
            curr_node = self.res[i]
            curr_node.left = None
            curr_node.right = self.res[i + 1]
        return self.res

# 105. 从前序与中序遍历序列构造二叉树
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        """ 前序：第一个到左面最后一个。中序：左面最后一个到第一个 """
        root: Optional[TreeNode]
        def dfs(pre: List[int], inor: List[int]) -> Optional[TreeNode]:
            if len(pre) <= 0:
                return None
            node = TreeNode(val=pre[0])
            first_node = pre[0]
            index: int
            for i, in_val in enumerate(inor):
                if in_val == first_node:
                    index = i
            node.left = dfs(pre[1 : index + 1], inor[0 : index])
            node.right = dfs(pre[index + 1:], inor[index + 1:])
            return node
        return dfs(preorder, inorder)

# 236. 二叉树的最近公共祖先 （后续遍历）
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        def dfs(node: TreeNode, p: TreeNode, q: TreeNode) -> Optional[TreeNode]:
            if not node:
                return None
            if node == p or node == q:
                return node
            left = dfs(node.left, p, q)
            right = dfs(node.right, p, q)
            if left and right:
                return node
            elif left:
                return left
            else:
                return right
        return dfs(root, p, q)

# 124. 二叉树中的最大路径和
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.max_num = -2**32 - 1
        def dfs(node: Optional[TreeNode]) -> int:
            if not node:
                return 0
            left = dfs(node.left)
            right = dfs(node.right)
            self.max_num = max(self.max_num, node.val + left + right)
            return max(0, max(left, right) + node.val)
        dfs(root)
        return self.max_num



# 662. 二叉树最大宽度
    def widthOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        p = [TreeIndex(index=1, node=root)]
        max_len = 0
        while p:
            max_len = max(max_len, p[len(p) - 1].index - p[0].index + 1)
            tmp = p
            p = []
            while tmp:
                curr = tmp[0]
                tmp = tmp[1:]
                if curr.node.left:
                    p.append(TreeIndex(index=2*curr.index, node=curr.node.left))
                if curr.node.right:
                    p.append(TreeIndex(index=2*curr.index + 1, node=curr.node.right))
        return max_len

# LCR 025. 两数相加 II
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        """将两个链表反转"""
        l1_re = self.reverseNode(l1)
        l2_re = self.reverseNode(l2)
        dum = ListNode(val=-1)
        cur = dum
        tmp = 0
        while l1_re or l2_re:
            l1_val = l1_re.val if l1_re else 0
            l2_val = l2_re.val if l2_re else 0
            l1_re = l1_re.next if l1_re else None
            l2_re = l2_re.next if l2_re else None
            num = (l1_val + l2_val + tmp) % 10
            tmp = (l1_val + l2_val + tmp) / 10
            cur.next = ListNode(val=num)
            cur = cur.next
        if tmp > 0:
            cur.next = ListNode(val=tmp)
        return self.reverseNode(dum.next)


    def reverseNode(self, node: Optional[ListNode]) -> ListNode:
        pre = None
        while node:
            node_next = node.next
            node.next = pre
            pre = node
            node = node_next
        return pre

# 面试题 02.06. 回文链表
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        l = []
        while head:
            l.append(head)
            head = head.next
        i, j = 0, len(l) - 1
        while i < j:
            if l[i].val != l[j].val:
                return False
            i += 1
            j -= 1
        return True
#5. 最长回文子串
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = True
        max_str = s[0]
        for L in range(2, n + 1):
            for i in range(n - L + 1):
                j = L + i - 1
                if L == 2:
                    dp[i][j] = s[i] == s[j]
                else:
                    dp[i][j] = dp[i + 1][j - 1] & (s[i] == s[j])
                if dp[i][j]:
                    max_str = s[i : j + 1]
        return max_str

#39. 组合总和
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res, path = [], []
        def dfs(index: int, left: int):
            if left < 0:
                return
            if left == 0:
                res.append(path.copy())
            for i in range(index, len(candidates)):
                path.append(candidates[i])
                dfs(i, left - candidates[i])
                path.pop()
        dfs(0, target)
        return res

#40. 组合总和 II
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        path = []
        res = []
        candidates.sort()
        def dfs(index: int, left: int):
            if left < 0:
                return
            if left == 0:
                res.append(path.copy())
            for i in range(index, len(candidates)):
                if i > index and candidates[i] == candidates[i - 1]:
                    continue
                path.append(candidates[i])
                dfs(i + 1, left - candidates[i])
                path.pop()
        dfs(0, target)
        return res

#113. 路径总和 II
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        res = []
        path = []
        def dfs(node: Optional[TreeNode], remain: int):
            if not node:
                return
            remain -= node.val
            path.append(node.val)
            if remain == 0 and node.left is None and node.right is None:
                res.append(path.copy())
                path.pop()
                return
            dfs(node.left, remain)
            dfs(node.right, remain)
            path.pop()
        dfs(root, targetSum)
        return res

#53. 最大子数组和
    def maxSubArray(self, nums: List[int]) -> int:
        curr, maxnum = nums[0], nums[0]
        for i in range(1, len(nums)):
            curr = max(nums[i], curr + nums[i])
            maxnum = max(curr, maxnum)
        return maxnum

#55. 跳跃游戏
    def canJump(self, nums: List[int]) -> bool:
        pos = 0
        for i in range(0, len(nums)):
            if pos < i:
                return False
            pos = max(pos, i + nums[i])
        return True

#64. 最小路径和
    def minPathSum(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        dp = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if i == j == 0:
                    dp[i][j] = grid[0][0]
                elif i == 0:
                    dp[i][j] = dp[i][j - 1] + grid[i][j]
                elif j == 0:
                    dp[i][j] = dp[i - 1][j] + grid[i][j]
                else:
                    dp[i][j] = min(dp[i][j - 1], dp[i - 1][j]) + grid[i][j]
        return dp[m - 1][n - 1]

#435. 无重叠区间
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda x:x[1])
        m = len(intervals)
        res = 0
        pivot = intervals[0][1]
        for i in range(1, m):
            if pivot > intervals[i][0]:
                res += 1
            else:
                pivot = intervals[i][1]
        return res


#121. 买卖股票的最佳时机
    def maxProfit(self, prices: List[int]) -> int:
        min_price, max_profit = prices[0], 0
        for i in range(len(prices)):
            if prices[i] > min_price:
                max_profit = max(max_profit, prices[i] - min_price)
            else:
                min_price = prices[i]
        return max_profit

#122. 买卖股票的最佳时机 II
    def maxProfit(self, prices: List[int]) -> int:
        """持有 or 未持有"""
        dp = [[0] * 2 for _ in range(len(prices))]
        dp[0][0] = -prices[0]
        for i in range(1, len(prices)):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] - prices[i])
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] + prices[i])
        return dp[len(prices) - 1][1]

#123. 买卖股票的最佳时机 III
    def maxProfit(self, prices: List[int]) -> int:
        dp = [[0] * 5 for _ in range(len(prices))]
        """ 未持有/持有第一支/卖出第一支/持有第二支/卖出第二支 """
        dp[0][1], dp[0][3]= -prices[0], -prices[0]
        for i in range(1, len(prices)):
            dp[i][0] = dp[i - 1][0]
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
            dp[i][2] = max(dp[i - 1][2], dp[i - 1][1] + prices[i])
            dp[i][3] = max(dp[i - 1][3], dp[i - 1][2] - prices[i])
            dp[i][4] = max(dp[i - 1][4], dp[i - 1][3] + prices[i])
        return max(dp[len(prices) - 1])

#309. 买卖股票的最佳时机含冷冻期
    def maxProfit(self, prices: List[int]) -> int:
        """持有/未持有/未持有（有卖出）"""
        dp = [[0] * 3 for _ in range(len(prices))]
        dp[0][0] = -prices[0]
        for i in range(1, len(prices)):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] - prices[i])
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][2])
            dp[i][2] = dp[i - 1][0] + prices[i]
        return max(dp[len(prices) - 1][1], dp[len(prices) - 1][2])

#714. 买卖股票的最佳时机含手续费
    def maxProfit(self, prices: List[int], fee: int) -> int:
        dp = [[0] * 2 for _ in range(len(prices))]
        """持有/未持有"""
        dp[0][0] = -prices[0]
        for i in range(1, len(prices)):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] - prices[i])
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] + prices[i] - fee)
        return dp[len(prices) - 1][1]


#395. 至少有 K 个重复字符的最长子串
    def longestSubstring(self, s: str, k: int) -> int:
        n = len(s)
        max_len = 0
        for i in range(n):
            dic = {}
            num = 0
            for j in range(i, n):
                if s[j] not in dic:
                    dic[s[j]] = 0
                dic[s[j]] += 1
                if dic[s[j]] == k:
                    num += 1
                if num == len(dic):
                    max_len = max(max_len, j - i + 1)
        return max_len



#200. 岛屿数量
    def numIslands(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        def dfs(i:int, j:int):
            if i < 0 or j < 0 or i > m - 1 or j > n - 1:
                return
            if grid[i][j] == '0' or grid[i][j] == '2':
                return
            grid[i][j] = '2'
            dfs(i - 1, j)
            dfs(i + 1, j)
            dfs(i, j - 1)
            dfs(i, j + 1)

        self.res = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    self.res += 1
                    dfs(i, j)
        return self.res

#124. 二叉树中的最大路径和
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        def dfs(node:Optional[TreeNode]) -> int:
            if not node:
                return 0
            left = dfs(node.left)
            right = dfs(node.right)
            self.max_sum = max(self.max_sum, left + right + node.val)
            return max(0, max(left, right) + node.val)
        dfs(root)
        return self.max_sum

##124. 二叉树中的最大路径和
    def maxPathSum2(self, root: Optional[TreeNode]) -> int:
        def dfs(node: Optional[TreeNode]) -> int:
            if not node:
                return 0
            left = dfs(node.left)
            right = dfs(node.right)
            self.max_num = max(self.max_num, left + right + node.val)
            return max(0, max(left, right) + node.val)
        dfs(root)
        return self.max_num

#827. 最大人工岛
    def largestIsland(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        def dfs(i: int, j: int, index: int) -> int:
            if i <0 or j < 0 or i > m - 1 or j > n - 1:
                return 0
            if grid[i][j] == 0 or grid[i][j] > 1:
                return 0
            grid[i][j] = index
            return 1 + dfs(i + 1,j) + dfs(i - 1, j) + dfs(i, j - 1) + dfs(i, j + 1)

        index = 2
        p_dic = defaultdict(int)
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    p = dfs(i, j, index)
                    p_dic[index] = p
                    index += 1
        max_m = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0:
                    userd = []
                    """up down left right"""
                    up, down, left, right = 0, 0, 0, 0
                    if i - 1 >= 0:
                        up = p_dic[grid[i - 1][j]]
                        userd.append(grid[i - 1][j])
                    if i + 1 < m:
                        if grid[i + 1][j] not in userd:
                            down = p_dic[grid[i + 1][j]]
                    if j - 1 >= 0:
                        if grid[i][j - 1] not in userd:
                            left = p_dic[grid[i][j - 1]]
                    if j + 1 < n:
                        if grid[i][j + 1] not in userd:
                            right = p_dic[grid[i][j + 1]]
                    max_m = max(max_m, up + down + left + right + 1)
        return max_m


    def minPathSum(self, grid: List[List[int]]) -> int:

        m, n = len(grid), len(grid[0])
        f = [[1] + [0] * (n - 1) for _ in range(m - 1)]

        dp = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    dp[i][j] = grid[i][j]
                elif i == 0:
                    dp[i][j] = dp[i][j - 1] + grid[i][j]
                elif j == 0:
                    dp[i][j] = dp[i - 1][j] + grid[i][j]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
        return dp[m - 1][n - 1]

#207. 课程表
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        dic = defaultdict(list)
        indeg = {i: 0 for i in range(numCourses)}
        for pre in prerequisites:
            dic[pre[1]].append(pre[0])
            indeg[pre[0]] += 1
        p = collections.deque([key for key, value in indeg.items() if value == 0])

        res = 0
        while p:
            res += 1
            u = p.popleft()
            for v in dic[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    p.append(v)
        return res == numCourses

#210. 课程表 II
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        """构建图，入度"""
        dic = defaultdict(list)
        indeg = {i:0 for i in range(numCourses)}
        for item in prerequisites:
            dic[item[1]].append(item[0])
            indeg[item[0]] += 1

        """入度为0"""
        p = collections.deque([key for key, value in indeg.items() if value == 0])
        res = []
        while p:
            course = p.popleft()
            res.append(course)
            for c in dic[course]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    p.append(c)
        return res

#124. 二叉树中的最大路径和
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.max_num = -2 ** 31
        def dfs(node: Optional[TreeNode]) -> int:
            if not node:
                return 0
            left = dfs(node.left)
            right = dfs(node.right)
            self.max_num = max(self.max_num, left + right + node.val)
            return max(0, max(left, right) + node.val)
        dfs(root)
        return self.max_num



#39. 组合总和
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        path = []
        res = []
        def dfs(index: int, remain: int):
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
        path, res = [], []
        candidates.sort()
        def dfs(index: int, remain: int):
            if remain < 0:
                return
            if remain == 0:
                res.append(path.copy())

            for i in range(index, len(candidates)):
                if i > index and candidates[i] == candidates[i - 1]:
                    continue
                path.append(candidates[i])
                dfs(i + 1, remain - candidates[i])
                path.pop()
        dfs(0, target)
        return res


#112. 路径总和
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        def dfs(node: Optional[TreeNode], remain: int):
            if not node:
                return False
            if remain - node.val == 0 and not node.left and not node.right:
                return True
            return dfs(node.left, remain - node.val) or dfs(node.right, remain - node.val)
        return dfs(root, targetSum)
#113. 路径总和 II
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        path, res = [], []
        def dfs(node: Optional[TreeNode], remain: int):
            if not node:
                return
            path.append(node.val)
            if remain - node.val == 0 and not node.left and not node.right:
                res.append(path.copy())
                path.pop()
                return
            remain = remain - node.val
            dfs(node.left, remain)
            dfs(node.right, remain)
            path.pop()
        dfs(root, targetSum)
        return res

#5. 最长回文子串
    def longestPalindrome(self, s: str) -> str:
        n = len(str)
        dp = [[False] * n for _ in range(n) ]
        for i in range(n):
            dp[i][i] = True
        max_len = s[0]
        for l in range(2, n + 1):
            for i in range(n):
                j = l + i - 1
                if j >= n:
                    break
                flag = s[i] == s[j]
                if l == 2:
                    dp[i][j] = flag
                else:
                    dp[i][j] = flag & dp[i + 1][j - 1]
                if dp[i][j]:
                    max_len = s[i:j + 1]
        return max_len

#64. 最小路径和
    def minPathSum(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        dp = [[0] * n for _ in range(m)]
        dp[0][0] = grid[0][0]
        for i in range(m):
            for j in range(n):
                if i == j == 0:
                    continue
                elif i == 0:
                    dp[i][j] = grid[i][j] + dp[i][j - 1]
                elif j == 0:
                    dp[i][j] = grid[i][j] + dp[i - 1][j]
                else:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
        return dp[m - 1][n - 1]

#435. 无重叠区间
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda x: x[1])
        point = intervals[1]
        n = len(intervals)
        res = 0
        for i in  range(1, n):
            interval = intervals[i]
            if point > interval[0]:
                res += 1
            else:
                point = interval[1]
        return res

#53. 最大子数组和
    def maxSubArray(self, nums: List[int]) -> int:
        max_sum, max_curr, n = nums[0], nums[0], len(nums)
        for i in range(1, n):
            max_curr = max(nums[i], max_curr + nums[i])
            max_sum = max(max_sum, max_curr)
        return max_sum

#300. 最长递增子序列
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [1] * n
        for i in range(1, n):
            for j in range(0, i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[j] + 1, dp[i])
        return max(dp)

#1885. 统计数对
    def countPairs(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        diff = sorted([nums1[i] - nums2[i] for i in range(n)])
        l, r = 0, n - 1
        res = 0
        while l < r:
            num = diff[l] + diff[r]
            if num > 0:
                res += r - l
                r -= 1
            else:
                l += 1
        return res



class TreeIndex:
    def __init__(self, node: Optional[TreeNode], index: int):
        self.node = node
        self.index = index




if __name__ == "__main__":
    s = Solution()

