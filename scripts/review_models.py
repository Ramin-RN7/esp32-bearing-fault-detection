#!/usr/bin/env python3
"""Reproduce the archive review on a host, including its known feature mismatch.

Run from the repository root after installing requirements.txt. This loads trusted
archived joblib bundles, reconstructs the missing split, and scores the nine test
recordings without retraining. A NumPy shim exercises arithmetic only. The output
does not validate ulab compatibility or ESP32 execution. Generated files go to
artifacts/technical_review/ by default. See docs/TECHNICAL_REVIEW.md.
"""
import sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
import ast, json, os, subprocess, types, importlib.util, warnings, time
import numpy as np
import scipy.signal as sig
from scipy.stats import skew,kurtosis
import scipy
import librosa
import soundfile as sf
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score, matthews_corrcoef
import argparse, math, random
ROOT=BASE
OUT=BASE/'artifacts'/'technical_review'

def load_functions(path, namespace):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),namespace)
    return namespace

def reconstruct_splits():
    assignment={}
    for cls,folder in [('normal','normal'),('anomaly','fault')]:
        files=sorted((ROOT/'01_uottawa_wavs/wav'/folder).glob('*.wav'))
        random.Random(42).shuffle(files)
        n=len(files);nt=math.floor(.70*n);nv=max(1,round(.15*n))
        for i,p in enumerate(files):
            assignment[p.name]={'class':cls,'label':int(cls=='anomaly'),
                'split':'train' if i<nt else 'val' if i<nt+nv else 'test',
                'bearing_id':int(p.name.split('_')[1]),'path':p.relative_to(ROOT).as_posix()}
    return assignment,{'assumption':'Reconstructed from sorted filenames and seed 42. Original manifests are absent.'}

def metrics(y,pred):
    return {'confusion_matrix_normal_fault':confusion_matrix(y,pred,labels=[0,1]).tolist(),
        'accuracy':float(accuracy_score(y,pred)),
        'fault_f1':float(f1_score(y,pred,zero_division=0)),
        'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
        'mcc':float(matthews_corrcoef(y,pred))}

def functions(rel, **constants):
    ns=dict(np=np,sig=sig,librosa=librosa,sf=sf,skew=skew,kurtosis=kurtosis,Path=Path,
            butter=sig.butter,filtfilt=sig.filtfilt,resample_poly=sig.resample_poly,
            SR=16000,WIN_S=2.,BP_LO=1000.,BP_HI=5000.,BP_ORDER=4)
    tree=ast.parse((ROOT/rel).read_text(encoding='utf-8-sig'))
    for n in tree.body:
        if isinstance(n,ast.Assign):
            try:v=ast.literal_eval(n.value)
            except (ValueError,TypeError):continue
            for target in n.targets:
                if isinstance(target,ast.Name):ns[target.id]=v
    ns.update(constants)
    return load_functions(ROOT/rel,ns)

def module_from_path(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def feature_fft(x,fs,N):
    # Mathematical equivalent of the archive's Hann -> DFT -> bin pooling.
    buf=np.zeros(N,dtype=float); a=np.asarray(x,dtype=float)
    buf[-min(len(a),N):]=a[-N:];buf-=buf.mean();buf*=np.hanning(N)
    k1=max(1,int(1000*N/fs));k2=min(N//2,int(5000*N/fs))
    power=np.abs(np.fft.rfft(buf)[k1:k2+1])**2/N
    size,rem=divmod(len(power),25);start=0;out=[]
    for b in range(25):
        stop=start+size+int(b<rem);out.append(float(np.mean(power[start:stop])) if stop>start else 0.);start=stop
    return np.asarray(out)

def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT,help='Directory for diagnostic outputs. Archived sources are only read.')
    args=parser.parse_args();OUT=args.out.resolve();OUT.mkdir(parents=True,exist_ok=True)
    if __import__('sklearn').__version__!='1.7.1':
        raise SystemExit('Use scikit-learn==1.7.1, matching the archived models. See requirements.txt.')
    assignment,split=reconstruct_splits()
    a8=functions('03_feature_extraction/a8_bandpass_1_5k.py')
    agg8=functions('04_model_training_ablation/a35a_aggregate_A8_GMM_per_file.py')
    a16=functions('04_model_training_ablation/a16_env_log_sr8k_pca8_minmax.py')
    agg16=functions('04_model_training_ablation/a35b_aggregate_A16_MLP_per_file.py')
    # The original prep functions are used to retain PCM write/read quantization.
    stage=OUT/'reconstructed_stage';stage.mkdir(exist_ok=True)
    prep=functions('02_data_segmentation/prep_segments.py',ROOT=stage)
    bundles={n:joblib.load(ROOT/'05_model_export'/f'{n}.joblib') for n in ('gmm_A8','mlp_A16')}
    export_dir=OUT/'exported_A8'
    proc=subprocess.run([sys.executable,str(ROOT/'05_model_export/a47_export_gmm_json.py'),
                         '--bundle',str(ROOT/'05_model_export/gmm_A8.joblib'),'--out',str(export_dir)],
                         env=os.environ.copy(),capture_output=True,text=True,check=True)
    print(proc.stdout,flush=True)
    pre=json.loads((export_dir/'preprocess.json').read_text());gm=json.loads((export_dir/'gmm.json').read_text())
    # NumPy compatibility shim. This exercises the archived arithmetic, not ulab or a board.
    ulab=types.ModuleType('ulab');ulab.numpy=np;sys.modules['ulab']=ulab;sys.modules['ujson']=json
    rt=module_from_path('audit_model_rt',ROOT/'04_model_training_ablation/model_rt_gmm.py')
    dev=module_from_path('audit_feat_a8',ROOT/'03_feature_extraction/feat_a8.py')
    file_records=[];segment_records=[];device_records=[];parity_errors=[];parity_disagreements=[];raw_features=[]
    for name,row in sorted(assignment.items()):
        if row['split']!='test':continue
        standard=stage/'wav16k'/row['class']/name
        prep['standardize'](ROOT/row['path'],standard)
        class ManifestSink:
            def writerow(self,row):pass
        prep['segment_file'](standard,stage/'segments',name,row['class'],ManifestSink())
        paths=sorted((stage/'segments'/row['class']).glob(Path(name).stem+'__seg*.wav'))
        logs={k:[] for k in ('A8_train_features','A8_file_features','A16_train_features','A16_file_features')}
        for p in paths:
            s=a8['load_wav'](p)
            f8=a8['features25'](a8['bandpass'](s,16000),16000)
            f8_file=agg8['features25'](agg8['bandpass_1_5k'](s),16000)
            s8=a16['load_wav'](p)
            f16=a16['features25'](a16['bp_hilbert_envelope_log'](s8),8000)
            f16_file=agg16['features25'](agg16['bp_hilbert_envelope_log'](s8),8000)
            raw_features.append(f8)
            vals={}
            for label,feat in [('A8_train_features',f8),('A8_file_features',f8_file)]:
                b=bundles['gmm_A8'];z=b['pca'].transform(b['scaler'].transform(feat[None,:]));ll=float(b['gmm'].score_samples(z)[0]);vals[label]=ll
                lr=float(rt.gmm_loglik(rt.preprocess_vec(feat,pre),gm));parity_errors.append(abs(lr-ll))
                if (lr<gm['thr']) != (ll<gm['thr']):parity_disagreements.append({'recording':name,'segment':p.name})
            for label,feat in [('A16_train_features',f16),('A16_file_features',f16_file)]:
                b=bundles['mlp_A16'];z=b['pca'].transform(b['scaler'].transform(feat[None,:]));vals[label]=float(b['mlp'].predict_proba(z)[0,1])
            for k,v in vals.items():logs[k].append(v)
            segment_records.append({'recording':name,'segment':p.name,'label':row['label'],**vals})
            for N in (256,512,1024):
                fd=feature_fft(s,16000,N);z=(fd-bundles['gmm_A8']['scaler'].mean_)/(bundles['gmm_A8']['scaler'].scale_+1e-8)
                ll=float(rt.gmm_loglik(rt.preprocess_vec(fd,pre),gm))
                device_records.append({'recording':name,'segment':p.name,'N':N,'label':row['label'],
                                       'max_abs_training_zscore':float(np.max(np.abs(z))),
                                       'log_likelihood':ll,'device_normal_label':int(ll>=gm['thr'])})
        file_records.append({'recording':name,'label':row['label'],**{k:float(np.mean(v)) for k,v in logs.items()}})
        print('Scored',name,len(paths),'segments',flush=True)
    # Confirm the fast diagnostic feature formula agrees with the actual archived DFT fallback.
    test_input=np.random.default_rng(42).normal(size=256).tolist()
    reference=np.asarray(dev.feat25(test_input,16000,256))
    feat_error=float(np.max(np.abs(reference-feature_fft(test_input,16000,256))))
    # Exercise predict_gmm's actual loader calls for a single prediction.
    loads=[]
    def jload(path):
        loads.append(path);return pre if path.endswith('preprocess.json') else gm
    rt.jload=jload;rt.predict_gmm([0.0]*25)
    headline={};segment_metrics={}
    for key in ('A8_train_features','A8_file_features','A16_train_features','A16_file_features'):
        threshold=gm['thr'] if key.startswith('A8') else .5
        pred=lambda v:int(v<threshold) if key.startswith('A8') else int(v>=threshold)
        headline[key]=metrics([x['label'] for x in file_records],[pred(x[key]) for x in file_records])
        segment_metrics[key]=metrics([x['label'] for x in segment_records],[pred(x[key]) for x in segment_records])
    report={'environment':{'numpy':np.__version__,'scipy':scipy.__version__,'librosa':librosa.__version__,'soundfile':sf.__version__,
                           'sklearn':__import__('sklearn').__version__},
            'scope':'Host reconstruction using archived model bundles and archived feature functions. Original manifests and complete firmware are absent. No retraining or hardware timing.',
            'file_predictions':file_records,'file_metrics':headline,'segment_metrics':segment_metrics,
            'gmm_threshold':gm['thr'],'gmm_runtime_vs_sklearn_max_abs_loglik_error':max(parity_errors),
            'gmm_runtime_vs_sklearn_predictions':len(parity_errors),
            'gmm_runtime_vs_sklearn_decision_disagreements':parity_disagreements,
            'hardware_execution_verified':False,
            'device_feature_sampling':{'frames_checked':len(device_records),'fault_predictions':sum(x['device_normal_label']==0 for x in device_records),
                'normal_labeled_frames':sum(x['label']==0 for x in device_records),
                'max_abs_training_zscore_range':[min(x['max_abs_training_zscore'] for x in device_records),max(x['max_abs_training_zscore'] for x in device_records)],
                'loglik_range':[min(x['log_likelihood'] for x in device_records),max(x['log_likelihood'] for x in device_records)],
                'fast_fft_formula_vs_archive_dft_max_error':feat_error},
            'json_loads_per_predict_call':loads,
            'a16_train_filter_high_hz':3500,'a16_file_filter_high_hz':3600}
    (OUT/'model_diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'segment_predictions.json').write_text(json.dumps(segment_records,indent=2)+'\n')
    (OUT/'device_feature_diagnostics.json').write_text(json.dumps(device_records,indent=2)+'\n')
    np.save(OUT/'a8_test_features.npy',np.asarray(raw_features))
    print(json.dumps({k:v for k,v in report.items() if k!='file_predictions'},indent=2),flush=True)

if __name__=='__main__':main()
