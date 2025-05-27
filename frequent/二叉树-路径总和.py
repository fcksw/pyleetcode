from typing import Optional, List

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class TreeIndex:
    def __init__(self, index, node):
        self.node = node
        self.index = index


class Solution:
    # 112. 路径总和
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:

        def dfs(node: Optional[TreeNode], remain: int) -> bool:
            if not node:
                return False
            if remain - node.val == 0 and not node.left and not node.right:
                return True
            return dfs(node.left, remain - node.val) or dfs(node.right, remain - node.val)
        return dfs(root, targetSum)

    # 113. 路径总和 II
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        path, res = [], []
        def dfs(node: Optional[TreeNode], remain:int):
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


    # 662. 二叉树最大宽度
    def widthOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        root_node = TreeIndex(1, root)
        p = [root_node]
        res = 0
        while p:
            res = max(p[len(p) - 1].index - p[0].index + 1, res)
            tmp = p
            p = []
            while tmp:
                first = tmp[0]
                tmp = tmp[1:]
                if first.node.left:
                    u = TreeIndex(first.index * 2, first.node.left)
                    p.append(u)
                if first.node.right:
                    u = TreeIndex(first.index * 2 + 1, first.node.right)
                    p.append(u)
        return res


