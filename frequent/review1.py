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


if __name__ == '__main__':
    pass