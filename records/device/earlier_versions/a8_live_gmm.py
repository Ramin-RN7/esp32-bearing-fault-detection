# /a8_live_gmm.py — capture 256 @16k → A8 features → GMM predict
import time, struct
import feat_a8
import model_rt_gmm as rt

def _fake_frame(n=256):
    try:
        import urandom
        return [(urandom.getrandbits(16)-32768)/32768 for _ in range(n)]
    except Exception:
        return [0.0]*n

def capture_frame(n=256, rate=16000):
    try:
        import M5
        from M5 import Mic
        M5.begin()
        Mic.begin()
        try: Mic.setSampleRate(rate)
        except Exception: pass
        buf = bytearray(n*2)                 # int16 mono buffer
        Mic.record(buf, rate, False)         # False=mono
        while Mic.isRecording():
            time.sleep_ms(5)
        Mic.end()
        vals = struct.unpack("<%dh" % n, buf)  # little-endian int16
        return [v/32768 for v in vals]
    except Exception:
        return _fake_frame(n)

def run_once():
    x = capture_frame(256, 16000)
    f = feat_a8.feat25(x, fs=16000, N=256)     # 25-D
    y, ll, thr = rt.predict_gmm(f)
    print("feat_len:", len(f))
    print("GMM:", y, ll, thr)

if __name__ == "__main__":
    run_once()