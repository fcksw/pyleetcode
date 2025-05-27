from typing import List, Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


    def maxStep(self, m, n) -> int:
        res = 0
        """ (m -1 ) * 2  =  m * 2 - 2  (趋近) ||  m * 2 - 1 """
        while m != n:
            if m * 2 < n:
                m = m * 2
                res += 1
                continue
            else:
                p = (m - 1) * 2
                q = m * 2 - 1
                if abs(p - n) > abs(q - n):
                    m = m * 2 -1
                else:
                    m = (m - 1) * 2

        return res



class Solution:
    pass



if __name__ == '__main__':
    pass




