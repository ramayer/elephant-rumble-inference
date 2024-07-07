from collections import deque

class TripleBufferedIterator:
    def __init__(self, iterable_or_iterator):
        self.iter = iter(iterable_or_iterator)
        self.buffer = deque(maxlen=3)
        self.__iter__()

    def __iter__(self):
        self.iter = self.iter.__iter__()
        self.buffer.clear()
        self.buffer.append(None)
        return self

    def __next__(self):
        while len(self.buffer) < 3:
            try:
                self.buffer.append(next(self.iter))
            except StopIteration:
                break
        if len(self.buffer) <2:
            raise StopIteration
        prv = self.buffer.popleft()
        cur = self.buffer[0]
        nxt = self.buffer[1] if len(self.buffer)>1 else None
        return prv,cur,nxt
