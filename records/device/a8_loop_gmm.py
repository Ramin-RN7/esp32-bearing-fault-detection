# /a8_loop_gmm.py — capture->features->GMM every ~300 ms (10 times)
import time
import feat_a8
import model_rt_gmm as rt
from a8_live_gmm import capture_frame  # reuse the mic/fallback capture

def run_loop(n_runs=10, period_ms=300):
    for i in range(n_runs):
        x = capture_frame(256, 16000)
        f = feat_a8.feat25(x, fs=16000, N=256)
        y, ll, thr = rt.predict_gmm(f)
        print("i:", i, "y:", y, "ll:", ll, "thr:", thr)
        time.sleep_ms(period_ms)

if __name__ == "__main__":
    run_loop()
