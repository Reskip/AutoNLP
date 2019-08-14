import time


class Timer():
    def __init__(self):
        self._timmers = dict()

    def start(self, key):
        self.check(key)
        self._timmers[key].start()

    def end(self, key):
        self._timmers[key].end()

    def get(self, key):
        self.check(key)
        return self._timmers[key]

    def check(self, key):
        if key not in self._timmers:
            self._timmers[key] = _Timer()


class _Timer():
    def __init__(self):
        self._cnt = 0
        self._total_time = 0.0
        self._start = None

    def start(self):
        self._start = time.time()

    def end(self):
        self._total_time += time.time() - self._start
        self._start = None
        self._cnt += 1

    def avg(self):
        if not self._cnt:
            return 0.0
        return self._total_time / self._cnt
