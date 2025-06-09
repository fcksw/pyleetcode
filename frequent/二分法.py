
import collections
from typing import List



class Solution:
    # 34. 在排序数组中查找元素的第一个和最后一个位置
    # [5,7,7,8,8,10], 8
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if not nums:
            return [-1, -1]
        left, right = 0, len(nums) - 1
        first = self.lower(nums, target, left, right)
        if first > right or nums[first] != target:
            return [-1, -1]
        second = self.lower(nums, target + 1, left, right) - 1
        return [first, second]

    def lower(self, nums: List[int], target:int, left: int, right: int):
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] >= target:
                right = mid - 1
            else:
                left = mid + 1
        return left


    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        dic = collections.defaultdict(list)
        indegre = {i: 0 for i in range(numCourses)}
        for pre in prerequisites:
            indegre[pre[0]] += 1
            dic[pre[1]].append(pre[0])
        p = collections.deque([key for key, value in indegre.items() if value == 0])

        res = 0
        while p:
            u = p.popleft()
            res += 1
            for course in dic[u]:
                indegre[course] -= 1
                if indegre[course] == 0:
                    p.append(course)
        return res == numCourses




if __name__ == "__main__":

    test1 = ['a', 'b', 'c']
    test2 = [1,2,3]
    for x in zip(test1, test2):
        print(x)
    pass