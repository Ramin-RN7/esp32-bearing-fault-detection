# /a8_eval_wav_gmm.py ? Evaluate A8->GMM on WAVs, log timing + RAM
import os, struct, time, gc
import feat_a8
import model_rt_gmm as rt

def _parse_wav_header(f):
    assert f.read(4) == b'RIFF'; f.read(4); assert f.read(4) == b'WAVE'
    sr=0; ch=0; bps=0; data_pos=None; data_size=0
    while True:
        cid = f.read(4)
        if not cid: break
        csz = struct.unpack('<I', f.read(4))[0]
        if cid == b'fmt ':
            fmt = f.read(csz)
            audio_fmt, ch, sr, byte_rate, block_align, bps = struct.unpack('<HHIIHH', fmt[:16])
            if csz & 1: f.read(1)
        elif cid == b'data':
            data_pos = f.tell(); data_size = csz
            break
        else:
            f.seek(csz + (csz & 1), 1)
    return sr, ch, bps, data_pos, data_size

def _frames_from_wav(path, N=256, hop=128):
    with open(path, 'rb') as f:
        sr, ch, bps, pos, size = _parse_wav_header(f)
        if not (sr==16000 and ch==1 and bps==16 and pos is not None):
            raise ValueError("need 16kHz mono 16-bit PCM")
        f.seek(pos)
        raw = f.read(size)
    n = size // 2
    ints = struct.unpack('<%dh' % n, raw)
    i = 0
    while i + N <= n:
        yield [ints[j]/32768 for j in range(i, i+N)]
        i += hop

def eval_one(path, N=256, hop=128):
    nfrm=0; sum_feat_us=0; sum_inf_us=0
    ys=[]; lls=[]
    for frm in _frames_from_wav(path, N, hop):
        t0 = time.ticks_us()
        feat = feat_a8.feat25(frm, fs=16000, N=N)
        t1 = time.ticks_us()
        y, ll, thr = rt.predict_gmm(feat)
        t2 = time.ticks_us()
        sum_feat_us += time.ticks_diff(t1, t0)
        sum_inf_us  += time.ticks_diff(t2, t1)
        ys.append(y); lls.append(ll); nfrm += 1
    if nfrm == 0:
        return None
    mean_y = 1 if (sum(ys)/nfrm) >= 0.5 else 0
    maj_y  = 1 if (sum(1 for v in ys if v==1) > nfrm/2) else 0
    mean_ll = sum(lls)/nfrm
    gc.collect()
    free_kb = gc.mem_free()//1024
    return {"frames": nfrm,"mean_y": mean_y,"maj_y": maj_y,"mean_ll": mean_ll,"thr": thr,
            "avg_feat_ms": (sum_feat_us/nfrm)/1000.0, "avg_infer_ms": (sum_inf_us/nfrm)/1000.0,
            "mem_free_kB": int(free_kb)}

def run_eval(dirpath="/wav", N=256, hop=128):
    try:
        files = ["/%s/%s"%(dirpath.strip('/'), f) for f in os.listdir(dirpath) if f.lower().endswith(".wav")]
    except Exception:
        files = []
    files.sort()
    if not files:
        print("No WAVs in", dirpath); return
    print("file,frames,mean_y,maj_y,mean_ll,thr,avg_feat_ms,avg_infer_ms,mem_free_kB")
    for p in files:
        try:
            r = eval_one(p, N, hop)
            if r is None: 
                print(p, "no_frames"); continue
            print("%s,%d,%d,%d,%.5f,%.5f,%.3f,%.3f,%d" % (
                p, r["frames"], r["mean_y"], r["maj_y"], r["mean_ll"], r["thr"],
                r["avg_feat_ms"], r["avg_infer_ms"], r["mem_free_kB"]))
        except Exception as e:
            print("%s,ERROR,%s" % (p, repr(e)))

if __name__ == "__main__":
    run_eval()
