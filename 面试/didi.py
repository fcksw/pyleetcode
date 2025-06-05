from typing import List

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
    def max_area(self, grip:List[List[int]]) -> int:
        m, n = len(grip), len(grip[0])
        res = 0
        for i in range(m):
            for j in range(n):
                if grip[i][j] == 1:
                    area = self.dfs(m, n, i, j, grip)
                    res = max(res, area)
        return res


    def dfs(self, m, n, i, j, grip):
        if i >= n or i < 0 or j >= m or j < 0:
            return 0
        if grip[i][j] == 0 or grip[i][j] == 2:
            return 0
        grip[i][j] = 2
        return self.dfs(m, n, i - 1, j ,grip) + self.dfs(m, n, i + 1, j ,grip) + self.dfs(m, n, i, j + 1 ,grip) + self.dfs(m, n, i, j - 1 ,grip) + 1




if __name__ == '__main__':
    pass