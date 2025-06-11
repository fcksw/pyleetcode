from typing import List
from math import inf

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
    """
    一个二维数组表示的矩阵，有0和1两种元素，0代表水，1代表陆地，相连的1代表岛屿，找出最大的岛屿面积
        输入：
        [1,0,1,1,1]
        [0,1,0,0,1]
        [1,1,1,1,0]
        [1,0,1,1,0]
    """
    def maxArea(self, grip:List[List[int]]) -> int:
        m, n = len(grip), len(grip[0])
        def dfs(i, j):
            if i < 0 or i >= m or j < 0 or j >= n:
                return 0
            if grip[i][j] != 1:
                return 0
            grip[i][j] = 2
            return dfs(i + 1, j) + dfs(i - 1, j) + dfs(i, j + 1) + dfs(i, j - 1) + 1
        res = 0
        for i in range(m):
            for j in range(n):
                if grip[i][j] == 1:
                    area = dfs(i, j)
                    res = max(area, res)
        return res


    # 判断一个二叉搜索树是否合法 未完全通过
    def isTrueTreeNode(self, root: TreeNode):
        def dfs(node: TreeNode) -> bool:
            if not node :
                return True
            left = node.left
            right = node.right
            res = True
            if left and left.val > node.val:
                res = False
            if right and right.val < node.val:
                res = False
            return dfs(node.left) and dfs(node.right) and res
        return dfs(root)

    # 判断一个二叉搜索树是否合法
    def isTrueTreeNode1(self, root: TreeNode):
        def dfs(node: TreeNode, max_inf, min_inf) -> bool:
            if not node:
                return True
            if node.val >= max_inf or node.val <= min_inf:
                return False
            return dfs(node.left, node.val, min_inf) and dfs(node.right, max_inf, node.val)
        return dfs(root, inf, -inf)



if __name__ == '__main__':
    pass