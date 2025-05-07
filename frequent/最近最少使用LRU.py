import collections
import datetime

"""
    LRU，增加过期超时，此处只使用惰性删除，不额外增加定时器进行定期清理
"""


class LinkNode:

    def __init__(self, pre, next, key, value, create_time):
        self.pre = pre
        self.next = next
        self.key = key
        self.value = value
        self.create_time = create_time

class LRUCache:

    __out_time = datetime.timedelta(minutes=10)

    def __init__(self, capacity: int, time):
        self.capacity = capacity
        self.size = 0
        self.head = LinkNode(None, None, 0, 0, time)
        self.tail = LinkNode(None, None, 0, 0, time)
        self.cache:[int, LinkNode] = collections.defaultdict()
        self.head.next = self.tail
        self.tail.pre = self.head


    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        # 判断是否过期（惰性删除）
        if node.create_time + self.__out_time < datetime.datetime.now():
            self.__del_node(node)
            return -1
        self.__del_node(node)
        self.__insert_node(node)
        return node.value



    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            self.__del_node(node)
            self.__insert_node(node)
            node.value = value
        else:
            node = LinkNode(None, None, key, value, datetime.datetime.now())
            self.__insert_node(node)



    def __del_node(self, node: LinkNode):
        """" size-1 cache(del) """
        self.size -= 1
        del self.cache[node.key]
        node.pre.next = node.next
        node.next.pre = node.pre
        node.next = None
        node.pre = None


    def __insert_node(self, node):
        if self.size >= self.capacity:
            """删除尾部"""
            tail_node = self.tail.pre
            self.__del_node(tail_node)
        node.create_time = datetime.datetime.now()
        self.size += 1
        self.cache[node.key] = node
        node.pre = self.head
        node.next = self.head.next
        node.next.pre = node
        self.head.next = node
