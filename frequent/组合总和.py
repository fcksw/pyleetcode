from typing import List


class Solution:

    # 39. 组合总和
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        path, res = [], []
        def dfs(i: int, remain: int):
            if remain < 0:
                return
            if remain == 0:
                res.append(path.copy())
            for index in range(i, len(candidates)):
                path.append(candidates[index])
                dfs(index, remain - candidates[index])
                path.pop()
        dfs(0, target)
        return res

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        path, res = [], []
        def dfs(i: int, remain: int):
            if remain < 0:
                return
            if remain == 0:
                res.append(path.copy())
                return
            for index in range(i, len(candidates)):
                if index > i and candidates[index - 1] == candidates[index]:
                    continue
                path.append(candidates[index])
                dfs(index + 1, remain - candidates[index])
                path.pop()
        dfs(0, target)
        return res


