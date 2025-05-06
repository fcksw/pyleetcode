import collections
from typing import List

# 695. 岛屿的最大面积
def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
    m, n = len(grid), len(grid[0])
    def dfs(i:int, j:int)-> int:
        if i >= m or i < 0 or j >= n or j < 0:
            return 0
        if grid[i][j] != 1:
            return 0
        grid[i][j] = 2
        return 1 + dfs(i - 1, j) + dfs(i + 1, j) + dfs(i, j - 1) + dfs(j, j + 1)
    max_p = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                p = dfs(i, j)
                max_p = max(p, max_p)
    return max_p

    pass

# 200. 岛屿数量
def numIslands(self, grid: List[List[str]]) -> int:
    m, n = len(grid), len(grid[0])
    def dfs(i, j):
        if i >= m or i < 0 or j >= n or j < 0:
            return
        if grid[i][j] != 1:
            return
        grid[i][j] = 2
        dfs(i + 1, j)
        dfs(i - 1, j)
        dfs(i, j - 1)
        dfs(i, j + 1)
    res = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                dfs(i, j)
                res += 1
    return res


#827. 最大人工岛
def largestIsland(self, grid: List[List[int]]) -> int:
    m, n = len(grid), len(grid[0])
    def dfs(i, j, index) -> int:
        if i >= m or i < 0 or j >= n or j < 0:
            return 0
        if grid[i][j] != 1:
            return 0
        grid[i][j] = index
        return 1 + dfs(i - 1, j, index) + dfs(i + 1, j, index) + dfs(i, j - 1, index)+ dfs(i, j + 1, index)
    index = 2
    dic = collections.defaultdict(int)
    max_p = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                p = dfs(i, j, index)
                max_p = max(max_p, p)
                dic[index] = p
                index += 1
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 0:
                used = []
                p = 1
                if i >= 1:
                    index = grid[i - 1][j]
                    used.append(index)
                    p += dic[index]
                if i < m - 1 and grid[i + 1][j] not in used:
                    index = grid[i + 1][j]
                    used.append(index)
                    p += dic[index]
                if j >= 1 and grid[i][j - 1] not in used:
                    index = grid[i][j - 1]
                    used.append(index)
                    p += dic[index]
                if j < n - 1 and grid[i][j + 1] not in used:
                    index = grid[i][j + 1]
                    used.append(index)
                    p += dic[index]
                max_p = max(p, max_p)
    return max_p