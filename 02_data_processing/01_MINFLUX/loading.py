import numpy as np
from scipy import optimize

import noctiluca as nl

def find_timestep(npy):
    tim = npy['tim']
    tid = npy['tid']

    dt = np.diff(tim)[np.diff(tid) == 0] # exclude steps across different trajectories
    if np.any(dt == 0):
        raise ValueError("Found zero time lag!")

    # Numerics might be better if everything is O(1)
    scale = np.min(dt)
    dt = dt / scale
    mindt = np.min(dt)
    
    # Step 1: rough estimate through MSE
    def mse(step):
        ints = np.round(dt/step).astype(int)
        return np.sum((dt-step*ints)**2)

    res = optimize.minimize(mse, mindt,
                            bounds=[(mindt, np.inf)],
                           )
    if not res.success:
        print(res)
        raise RuntimeError

    step = res.x

    # Step 2: identify real integer steps
    udts = []
    Ns = []
    cur = 0.5*step
    while cur < np.max(dt):
        ind = (dt > cur) & (dt < cur+step)
        Ns.append(np.sum(ind))
        if Ns[-1] > 0:
            udts.append(np.mean(dt[ind]))
            cur = udts[-1] + 0.5*step
        else:
            udts.append(np.nan)
            cur += step
    udts = np.array(udts)
    Ns   = np.array(Ns)

    # Step 3: fit actual best lag time
    ind = ~np.isnan(udts)
    with np.errstate(divide='ignore'):
        sigma = 1/np.sqrt(Ns[ind]-1)
    res = optimize.curve_fit(lambda x, a: a*x,
                             np.arange(len(udts))[ind]+1,
                             udts[ind],
                             sigma=sigma,
                            )

    return res[0][0]*scale

def load_file(file, relpath, dt_dict=None, filename2tags=None):
    npy = np.load(file)
    dt = find_timestep(npy)
    
    tim = npy['tim']
    tid = npy['tid']
    loc = npy['itr'][:, -1]['loc'][:, :2] # (#loc, 2) (we ignore z, since data is 2D)
    
    utid = np.unique(tid)
    
    data = nl.TaggedSet()
    for my_tid in utid:
        ind = tid == my_tid
        my_tim = tim[ind]
        my_loc = loc[ind]
        
        lag = np.round(np.diff(my_tim)/dt).astype(int)
        t = np.insert(np.cumsum(lag), 0, 0)
        
        traj = nl.Trajectory(my_loc, t=t, tid=my_tid, dt=dt)
        data.add(traj)
    
    relfile = file.relative_to(relpath)
    data.addTags(f'file={str(relfile)}')
    if filename2tags is not None:
        data.addTags(filename2tags(relfile))
    
    if dt_dict is not None:
        dt_dict[str(relfile)] = dt
    
    return data