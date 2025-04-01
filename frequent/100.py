import collections
from typing import List, Dict


class Solution:

    #1. 两数之和
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dic = {}
        collections.defaultdict(int)
        for i in range(len(nums)):
            if target - nums[i] in dic:
                return [dic[target - nums[i]], i]
            dic[nums[i]] = i
        return []


    #49. 字母异位词分组
    #输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
    #输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        dic = collections.defaultdict(list)
        for str in strs:
            key = "".join(sorted(str))
            dic[key].append(str)
        return list(dic.values())

    #128. 最长连续序列
    def longestConsecutive(self, nums: List[int]) -> int:
        nums.sort(reverse=False)
        n = len(nums)
        curr, max_num = 0, 0
        for i in range(n):
            if i == 0 or nums[i] - nums[i - 1] == 1:
                curr += 1
            elif nums[i] == nums[i - 1]:
                pass
            else:
                curr = 1
            max_num = max(max_num, curr)
        return max_num


#11. 盛最多水的容器
    def maxArea(self, height: List[int]) -> int:
        n = len(height)
        left, right = 0, n - 1
        max_area = 0
        while left < right:
            max_area = max(max_area, min(height[left], height[right]) * (right - left))
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return max_area

    #15. 三数之和
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        n = len(nums)
        res = []
        for i in range(n - 2):
            """过滤"""
            if i > 0 and nums[i] == nums[i -1]:
                continue
            l, r = i + 1, n - 1
            while l < r:
                sum_num = nums[i] + nums[l] + nums[r]
                if sum_num == 0:
                    res.append([nums[i], nums[l], nums[r]])
                    while l < r and nums[r] == nums[r - 1]:
                        r -= 1
                    while l < r and nums[l] == nums[l + 1]:
                        l += 1
                    l += 1
                    r -= 1
                elif sum_num > 0:
                    r -= 1
                else:
                    l += 1
        return  res

#42. 接雨水
    def trap(self, height: List[int]) -> int:
        n = len(height)
        l, r = 0, n - 1
        l_max, r_max = 0, 0
        res = 0
        while l < r:
            l_max = max(l_max, height[l])
            r_max = max(r_max, height[r])
            if height[l] < height[r]:
                res += l_max - height[l]
                l += 1
            else:
                res += r_max - height[r]
                r -= 1
        return res

#3. 无重复字符的最长子串
    def lengthOfLongestSubstring(self, s: str) -> int:
        dic: Dict[str, int] = {}
        s_max = 0
        l, r = 0, 0
        while r < len(s):
            if s[r] in dic and dic[s[r]] >= l:
                l = dic[s[r]] + 1
            dic[s[r]] = r
            s_max = max(s_max, r - l + 1)
            r += 1
        return s_max

#438. 找到字符串中所有字母异位词
    def findAnagrams(self, s: str, p: str) -> List[int]:
        p_sort = "".join(sorted(p))
        n = len(p)
        l,r = 0, n - 1
        res = []
        while r < len(s):
            s_cut = "".join(sorted(s[l: r + 1]))
            if s_cut == p_sort:
                res.append(l)
            l += 1
            r += 1
        return res


 #560. 和为 K 的子数组
    def subarraySum(self, nums: List[int], k: int) -> int:
        pre_sum = []
        n = len(nums)
        pre_sum[0] = 0
        for i in range(n):
            """sum[i + 1] = s[i] + nums[i]"""
            pre_sum[i + 1] = pre_sum[i] + nums[i]

        dic = collections.defaultdict(int)
        res = 0
        """ sj - si = k """
        for sj in pre_sum:
            si = sj - k
            if si in dic:
                res += dic[si]
            dic[sj] += 1
        return res








