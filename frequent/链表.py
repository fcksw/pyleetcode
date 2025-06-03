from typing import Optional

class ListNode:
    def __init__(self, val, next):
        self.val = val
        self.next = next

class Solution:

    # 92. 反转链表 II
    # 输入：head = [1,2,3,4,5], left = 2, right = 4
    # 输出：[1,4,3,2,5]
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        dum = ListNode(val=0, next=head)
        prev, end = dum, dum
        for _ in range(left - 1):
            prev = prev.next
        for _ in range(right):
            end = end.next
        start = prev.next
        next_node = end.next
        end.next = None
        prev.next = self.reverse(start)
        start.next = next_node
        return dum.next

    def reverse(self, pre: Optional[ListNode]):
        prev = None
        while pre:
            next_node = pre.next
            pre.next = prev
            prev = pre
            pre = next_node
        return prev
