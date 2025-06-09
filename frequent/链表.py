from turtledemo.penrose import start
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

    # k个一组链表反转
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dum = ListNode(next=head, val=0)
        prev, end = dum, dum
        while end is not None:
            for _ in range(k):
                if end is None:
                    break
                end = end.next
            if end is None:
                break
            next_node = end.next
            end.next = None
            start_node = prev.next
            prev.next = self.reverse(start_node)
            start_node.next = next_node
            prev = start_node
            end = start_node
        return dum

    def reverse(self, pre: Optional[ListNode]):
        prev = None
        while pre:
            next_node = pre.next
            pre.next = prev
            prev = pre
            pre = next_node
        return prev



class LinkedNode:
    def __init__(self, key, value, next, pre):
        self.key = key
        self.value = value
        self.pre = pre
        self.next = next

class LRUCache:

    def __init__(self, cap):
        self.size = 0
        self.cap = cap
        self.cache = {}
        self.tail = LinkedNode(0, 0, None, None)
        self.head = LinkedNode(0, 0, None, None)
        self.head.next = self.tail
        self.tail.pre = self.head


    def get(self, key: int) -> int:
        """ 先验证是否存在，不存在返回-1 """
        if key not in self.cache:
            return -1
        """ 如果存在，需要将节点添加的链表头 """
        node = self.cache.get(key)
        """ 先删除， 后增加 """
        self.__del_node(node)
        self.__insert_node(node)
        return node.value


    def put(self, key: int, value: int):
        if key in self.cache:
            node = self.cache.get(key)
            node.value = value
            self.__del_node(node)
            self.__insert_node(node)
        else:
            node = LinkedNode(key, value, None, None)
            self.__insert_node(node)



    def __insert_node(self, node):
        if self.size >= self.cap:
            self.__del_node(self.tail.pre)
        self.cache[node.key] = node
        self.size += 1
        node.pre = self.head
        node.next = self.head.next
        node.next.pre = node
        self.head.next = node



    def __del_node(self, node: LinkedNode):
        self.size -= 1
        del self.cache[node.key]
        node.next.pre = node.pre
        node.pre.next = node.next
        node.next = None
        node.pre = None
