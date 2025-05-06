import collections
from typing import List

class Solution:

    # 207. 课程表
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        dic = collections.defaultdict(list)
        inDegree = {key : 0 for key in range(numCourses)}
        for pre in prerequisites:
            dic[pre[1]].append(pre[0])
            inDegree[pre[1]] += 1
        p = collections.deque([key for key, value in inDegree.items() if value == 0])
        res = 0
        while p:
            node = p.pop()
            res += 1
            for q in dic[node]:
                inDegree[q] -= 1
                if inDegree[q] == 0:
                    p.append(q)
        return res == numCourses
