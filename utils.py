import time
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print ('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed

# Range utils
rngx = lambda x,y,lim,st: ((_x, y) for _x in range(x, lim, st))
rngy = lambda x,y,lim,st: ((x, _y) for _y in range(y, lim, st))