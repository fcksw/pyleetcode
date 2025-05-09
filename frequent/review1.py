import collections
import copy
from typing import List, Optional


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class TreeIndex:
    def __init__(self, index: int, node: TreeNode):
        self.index = index
        self.node = node

class Solution:

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res, path = [], []
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

    def combinationSum1(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        res, path = [], []
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

    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        res, path = [], []
        def dfs(node: TreeNode, remain: int):
            if not node:
                return
            remain = remain - node.val
            path.append(node.val)
            if not node.left and not node.right and remain == 0:
                res.append(path.copy())
                path.pop()
                return
            dfs(node.left, remain)
            dfs(node.right, remain)
            path.pop()
        dfs(root, targetSum)
        return res


    def widthOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        treeIndex = TreeIndex(index=1, node=root)
        p = [treeIndex]
        max_length = 0
        while p:
            max_length = max(max_length, p[len(p) - 1].index - p[0].index + 1)
            tmp = p
            p = []
            while tmp:
                u = tmp[0]
                tmp = tmp[1:]
                if u.node.left:
                    p.append(TreeIndex(node=u.node.left, index=u.index * 2))
                if u.node.right:
                    p.append(TreeIndex(node=u.node.right, index=u.index * 2 + 1))
        return max_length

    # 1143. 最长公共子序列
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        dp = [[0] * n for _ in range(m)]
        for i in range(m):
            if text1[i] == text2[0]:
                dp[i][0] = 1
        for j in range(n):
            if text1[0] == text2[j]:
                dp[0][j] = 1
        for i in range(1, m):
            for j in range(1, n):
                if text1[i] == text2[j]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m - 1][n - 1]

    # 395. 至少有 K 个重复字符的最长子串
    def longestSubstring(self, s: str, k: int) -> int:
        n = len(s)
        res = 0
        for i in range(n):
            dic = collections.defaultdict(int)
            num = 0
            for j in range(i, n):
                dic[s[j]] += 1
                if dic[s[j]] == k:
                    num += 1
                if num == len(dic):
                    res = max(res, j - i + 1)
        return res







if __name__ == '__main__':
    pass