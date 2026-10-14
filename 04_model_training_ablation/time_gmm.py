# /time_gmm.py ? timing for GMM (and A8+GMM)
import time, gc
import model_rt_gmm as rt
import feat_a8

def infer_only(n=500):
    gc.collect()
    f = [0.0]*25
    t0 = time.ticks_us()
    for _ in range(n):
        rt.predict_gmm(f)
    dt = time.ticks_diff(time.ticks_us(), t0)
    print("GMM_infer_ms_avg=", dt/n/1000, "n=", n, "mem_free_kB=", gc.mem_free()//1024)

def end_to_end(N=512, n=20):
    gc.collect()
    x = [0.0]*N
    # features
    t0 = time.ticks_us()
    for _ in range(n):
        f = feat_a8.feat25(x, 16000, N)
    t1 = time.ticks_us()
    # inference
    for _ in range(n):
        rt.predict_gmm(f)
    t2 = time.ticks_us()
    print("A8_feat_ms_avg=", time.ticks_diff(t1,t0)/n/1000,
          "GMM_infer_ms_avg=", time.ticks_diff(t2,t1)/n/1000,
          "mem_free_kB=", gc.mem_free()//1024)
