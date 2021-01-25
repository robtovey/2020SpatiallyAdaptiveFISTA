import os, sys, numba

if hasattr(os, 'sched_setaffinity'): # some UNIX systems
    def getaffinity(pid=None):
        pid = os.getpid() if pid is None else pid
        return os.sched_getaffinity(pid)
        
    def setaffinity(pid=None, cores=None):
        pid = os.getpid() if pid is None else pid
        if cores is None:
            cores = range(os.cpu_count())
        elif type(cores) is int:
            cores = (cores, )
        else:
            cores = tuple(int(c) for c in cores)
        os.sched_setaffinity(pid, cores)
        numba.set_num_threads(len(cores))

elif sys.platform == 'win32':
    import win32process, win32con, win32process, win32api

    def getaffinity(pid=None):
        pid = os.getpid() if pid is None else pid
        flags = win32con.PROCESS_QUERY_INFORMATION
        pHandle = win32api.OpenProcess(flags, 0, pid)
        
        aff = win32process.GetProcessAffinityMask(pHandle)[0]
        aff = '{0:b}'.format(aff) # convert to pure binary
        cores = tuple(i for i,b in enumerate(reversed(aff)) if b=='1')
        return cores
        
    def setaffinity(pid=None, cores=None):
        pid = os.getpid() if pid is None else pid
        flags = win32con.PROCESS_QUERY_INFORMATION
        flags |= win32con.PROCESS_SET_INFORMATION
        pHandle = win32api.OpenProcess(flags, 0, pid)

        if cores is None:
            cores = range(os.cpu_count())
        elif type(cores) is int:
            cores = (cores, )
        else:
            cores = tuple(int(c) for c in cores)
    
        aff = int(sum(2**c for c in cores))
        win32process.SetProcessAffinityMask(pHandle, aff)
        numba.set_num_threads(len(cores))

else:
    def getaffinity(pid=None):
        raise NotImplementedError
    def setaffinity(pid=None,cores=None):
        raise NotImplementedError
