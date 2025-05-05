from typing import List

class Solution:


    #56. 合并区间
    """
    以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。
    请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。
    """
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])
        tmp = intervals[0]
        res = []
        for i in range(1, len(intervals)):
            inter = intervals[i]
            if tmp[1] < inter[0]:
                res.append(tmp)
                tmp = inter
            else:
                tmp[1] = max(tmp[1], inter[1])
        res.append(tmp)
        return res

    #57. 插入区间
    """
    给你一个 无重叠的 ，按照区间起始端点排序的区间列表 intervals，
    其中 intervals[i] = [starti, endi] 表示第 i 个区间的开始和结束，
    并且 intervals 按照 starti 升序排列。
    同样给定一个区间 newInterval = [start, end] 表示另一个区间的开始和结束。
    在 intervals 中插入区间 newInterval，使得 intervals 依然按照 starti 升序排列
    ，且区间之间不重叠（如果有必要的话，可以合并区间）。返回插入之后的 intervals。
    """
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        left, right = newInterval[0], newInterval[1]
        merge = False
        res = []
        for inter in intervals:
            if not merge:
                if right < inter[0]:
                    res.append([left, right])
                    res.append(inter)
                    merge = True
                elif left > inter[1]:
                    res.append(inter)
                else:
                    left = min(inter[0], left)
                    right = max(inter[1], right)
            else:
                res.append(inter)
        if not merge:
            res.append([left, right])
        return res


    #435. 无重叠区间
    """给定一个区间的集合 intervals ，其中 intervals[i] = [starti, endi] 
    。返回 需要移除区间的最小数量，使剩余区间互不重叠 。
    """
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda x: x[1])
        tmp = intervals[0]
        res = 0
        for i in range(1, len(intervals)):
            inter = intervals[i]
            if tmp[1] <= inter[0]:
                tmp = inter
            else:
                res += 1
        return res

