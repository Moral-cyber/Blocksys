"""Independent NumPy audit of frozen calibration arrays and deployment states.

No official dataset, neural-model import/forward/backward, network or ML training.
Original sources/results are read-only. Only local audit artifacts are written.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode=True
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUNS=('diagnostic_v1','confirm_p20261203_s20261211')
KINDS=('zero_anchor_bias','temperature_scaling','bcts','vector_scaling')
DIMS={'zero_anchor_bias':10,'temperature_scaling':1,'bcts':11,'vector_scaling':20}
LAMBDA=10.
LOWER=.001
COUNTERS=collections.Counter()
ERRORS=collections.defaultdict(float)
FORBIDDEN=[]

def audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        name=os.fsdecode(args[0])
        if '/dataset/' in name or 'test_batch' in name or name.endswith(('.pt','.pth','.h5','.hdf5')):
            FORBIDDEN.append(name);raise AssertionError('raw data/checkpoints forbidden')
    if event in ('socket.connect','socket.getaddrinfo'):
        raise AssertionError('network forbidden')
sys.addaudithook(audit)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def check(label,condition):
    assert bool(condition),label
    COUNTERS[label]+=1

def close(label,a,b,tol=1e-9):
    error=float(np.max(np.abs(np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64))))
    ERRORS[label]=max(ERRORS[label],error)
    check(label,error<=tol)


def anchor(kind):
    p=np.zeros(DIMS[kind])
    if kind in ('temperature_scaling','bcts'):p[0]=1.
    elif kind=='vector_scaling':p[:10]=1.
    return p


def mapping(z,kind,t):
    z=np.asarray(z,dtype=np.float64);t=np.asarray(t,dtype=np.float64)
    if kind=='zero_anchor_bias':return z+t
    if kind=='temperature_scaling':return z*t[0]
    if kind=='bcts':return z*t[0]+t[1:]
    if kind=='vector_scaling':return z*t[:10]+t[10:]
    raise ValueError(kind)


def jgh(z,y,kind,t,derivatives=True):
    """Block covariance formulas, without source solver/design tensor reuse."""
    z=np.asarray(z,dtype=np.float64);t=np.asarray(t,dtype=np.float64)
    s=mapping(z,kind,t);mx=np.max(s,axis=1)
    ex=np.exp(s-mx[:,None]);p=ex/ex.sum(axis=1)[:,None]
    n=len(y);error=p.copy();error[np.arange(n),y]-=1.
    delta=t-anchor(kind)
    j=float(np.sum(mx+np.log(ex.sum(axis=1))-s[np.arange(n),y])+.5*LAMBDA*np.dot(delta,delta))
    if not derivatives:return j
    hb=np.diag(p.sum(axis=0))-p.T@p
    gb=error.sum(axis=0)
    if kind=='zero_anchor_bias':g,h=gb,hb
    elif kind in ('temperature_scaling','bcts'):
        ez=(p*z).sum(axis=1)
        ga=np.sum(error*z)
        haa=np.sum(p*z*z)-np.dot(ez,ez)
        if kind=='temperature_scaling':g,h=np.array([ga]),np.array([[haa]])
        else:
            hab=(p*(z-ez[:,None])).sum(axis=0)
            g=np.concatenate(([ga],gb));h=np.empty((11,11))
            h[0,0]=haa;h[0,1:]=hab;h[1:,0]=hab;h[1:,1:]=hb
    else:
        pz=p*z
        gaa=(error*z).sum(axis=0)
        haa=np.diag((p*z*z).sum(axis=0))-pz.T@pz
        hab=np.diag(pz.sum(axis=0))-pz.T@p
        g=np.concatenate((gaa,gb));h=np.block([[haa,hab],[hab.T,hb]])
    g=g+LAMBDA*delta;h=h+LAMBDA*np.eye(len(t))
    return j,g,(h+h.T)/2


def pg(t,g,kind):
    answer=g.copy()
    if kind in ('temperature_scaling','bcts') and t[0]<=LOWER+1e-12 and g[0]>0:answer[0]=0.
    return answer


def bias_of(t,kind):
    return None if kind=='temperature_scaling' else t if kind=='zero_anchor_bias' else t[1:] if kind=='bcts' else t[10:]


def stats(s,y):
    m=np.max(s,axis=1)
    nll=m+np.log(np.exp(s-m[:,None]).sum(axis=1))-s[np.arange(len(y)),y]
    return {'accuracy':int(np.count_nonzero(np.argmax(s,axis=1)==y))/len(y),
            'loss':float(np.sum(nll)/len(y)),'samples':int(len(y))}


def summarize(rows):
    n=np.array([r['holdout_samples'] for r in rows],dtype=np.float64)
    w=n/n.sum();answer={}
    for v in rows[0]['metrics']:
        a=np.array([r['metrics'][v]['accuracy'] for r in rows]);l=np.array([r['metrics'][v]['loss'] for r in rows])
        ordered=np.sort(a);position=(len(a)-1)*.1;left=int(np.floor(position));right=int(np.ceil(position))
        q=float(ordered[left]+(ordered[right]-ordered[left])*(position-left))
        answer[v]={'macro_accuracy':float(a.sum()/len(a)),'macro_loss':float(l.sum()/len(l)),
                   'weighted_accuracy':float(np.sum(a*w)),'weighted_loss':float(np.sum(l*w)),
                   'client_accuracy_q10':q}
    return answer


def independent_fit(z,y,kind):
    """Reduced centered-bias coordinates; exact one-bound quadratic Newton QP.

    No source Problem/fit calls. A different interior starting point is used.
    """
    base=anchor(kind);d=len(base)
    Q=np.linalg.qr(np.vstack((np.eye(9),-np.ones((1,9)))))[0]
    if kind=='temperature_scaling':R=np.eye(1)
    elif kind=='zero_anchor_bias':R=Q
    else:
        scale_count=1 if kind=='bcts' else 10
        R=np.zeros((d,scale_count+9));R[:scale_count,:scale_count]=np.eye(scale_count);R[scale_count:,scale_count:]=Q
    u=np.zeros(R.shape[1]);u[:1 if kind in ('temperature_scaling','bcts') else 10 if kind=='vector_scaling' else 0]=-.2
    history=[];reason='maximum_steps'
    for iteration in range(80):
        theta=base+R@u;j,g,h=jgh(z,y,kind,theta)
        gr=R.T@g;hr=R.T@h@R
        projected=gr.copy();bounded=kind in ('temperature_scaling','bcts')
        if bounded and theta[0]<=LOWER+1e-12 and gr[0]>0:projected[0]=0.
        norm=float(np.max(np.abs(projected)))
        history.append({'objective':j,'projected_gradient_inf':norm})
        if norm<1e-8:reason='projected_kkt';break
        step=-np.linalg.solve(hr,gr)
        if bounded and theta[0]+step[0]<LOWER:
            step[0]=LOWER-theta[0]
            if len(step)>1:step[1:]=np.linalg.solve(hr[1:,1:],-gr[1:]-hr[1:,0]*step[0])
        slope=float(gr@step);alpha=1.;accepted=False
        for _ in range(60):
            trial=u+alpha*step;t=base+R@trial
            if bounded and t[0]<LOWER-1e-13:alpha*=.5;continue
            trialj=jgh(z,y,kind,t,False)
            if slope<0 and trialj<=j+1e-4*alpha*slope:
                accepted=True;break
            alpha*=.5
        if not accepted:
            # A full Newton step within floating objective precision still
            # improves the stationarity of a strongly convex quadratic limit.
            trial=u+step;t=base+R@trial;trialj=jgh(z,y,kind,t,False)
            if norm<1e-5 and trialj<=j+16*np.finfo(float).eps*max(1,abs(j)):
                u=trial;reason='float64_resolution';break
            raise AssertionError('independent reduced QP Newton line search failed')
        u=trial
        if np.max(np.abs(alpha*step))<1e-14:reason='float64_resolution';break
    theta=base+R@u;j,g,h=jgh(z,y,kind,theta)
    return theta,{'objective':j,'projected_gradient_inf':float(np.max(np.abs(pg(theta,g,kind)))),
                  'iterations':iteration+1,'reason':reason,'objective_trace':[x['objective'] for x in history]}


def read_npz(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}


def main():
    began=time.perf_counter()
    runs={r:json.loads((HERE/r/'result.json').read_text()) for r in RUNS}
    protected={HERE/'PLAN.md',HERE/'solver.py',HERE/'probe.py'}
    for run,r in runs.items():
        protected.update([HERE/run/'result.json',HERE/run/'inputs_before.json',Path(r['input_root'])/'result.json'])
        for model in r['results']:
            for c in range(20):protected.update([HERE/run/f'{model}_client{c}.npz',Path(r['input_root'])/f'{model}_client{c}.npz'])
    before={str(p):sha(p) for p in sorted(protected)}
    # The original solver is imported ONLY for formula/finite-difference fixture
    # comparisons and two artificial boundary cases, never original run fitting.
    spec=importlib.util.spec_from_file_location('_read_only_calibration_solver',HERE/'solver.py')
    source_solver=importlib.util.module_from_spec(spec);sys.modules[spec.name]=source_solver;spec.loader.exec_module(source_solver)
    derivatives=[];rng=np.random.default_rng(20261104);z=rng.normal(size=(19,10));y=rng.integers(0,10,19)
    for kind in KINDS:
        t=anchor(kind)+rng.normal(0,.07,DIMS[kind]);problem=source_solver.Problem(z,y,kind)
        ours=jgh(z,y,kind,t);theirs=problem.objective(t)
        close('source_mapping_vs_independent',problem.score(t),mapping(z,kind,t),1e-12)
        close('source_objective_vs_independent',theirs[0],ours[0],1e-10)
        close('source_gradient_vs_independent',theirs[1],ours[1],1e-10)
        close('source_hessian_vs_independent',theirs[2],ours[2],1e-10)
        numericg=np.zeros(len(t));numerich=np.empty((len(t),len(t)))
        for j in range(len(t)):
            step=1e-5*(1+abs(t[j]));plus=t.copy();minus=t.copy();plus[j]+=step;minus[j]-=step
            numericg[j]=(jgh(z,y,kind,plus,False)-jgh(z,y,kind,minus,False))/(2*step)
            numerich[:,j]=(jgh(z,y,kind,plus)[1]-jgh(z,y,kind,minus)[1])/(2*step)
        close('finite_difference_gradient',numericg,theirs[1],2e-6)
        close('finite_difference_hessian',numerich,theirs[2],2e-6)
        check('synthetic_ridge_hessian_psd',np.linalg.eigvalsh(ours[2]).min()>=10-1e-10)
        derivatives.append({'kind':kind,'gradient_max_error':float(np.max(np.abs(numericg-theirs[1]))),
                            'hessian_max_error':float(np.max(np.abs(numerich-theirs[2])))})
    boundary=[]
    by=np.arange(100,dtype=np.int64)%10;bz=np.zeros((100,10));bz[np.arange(100),(by+1)%10]=10.
    for kind in ('temperature_scaling','bcts'):
        state,diagnostic=source_solver.fit(bz,by,kind)
        theta=np.array(diagnostic['parameters_float64']);j,g,h=jgh(bz,by,kind,theta)
        ours,opt=independent_fit(bz,by,kind)
        check('synthetic_positive_scale_lower_bound_active',abs(theta[0]-LOWER)<1e-12)
        check('synthetic_boundary_KKT_multiplier_nonnegative',g[0]>0)
        check('synthetic_boundary_projected_KKT',np.max(np.abs(pg(theta,g,kind)))<1e-7)
        close('independent_boundary_optimizer',theta,ours,1e-8)
        check('synthetic_boundary_state_positive',state[0]>0)
        check('synthetic_boundary_trace_monotone',all(a['objective']>=b['objective'] for a,b in zip(diagnostic['trace'],diagnostic['trace'][1:])))
        boundary.append({'kind':kind,'a':float(theta[0]),'gradient_a':float(g[0]),'projected_gradient_inf':float(np.max(np.abs(pg(theta,g,kind)))),
                         'independent_parameters_max_error':float(np.max(np.abs(theta-ours)))})
    summaries={};client_metrics={};state_audits=[];stop_counts=collections.Counter();strict=0;bound_count=0;maxkkt=0.;refits=[]
    fixed_refit={('diagnostic_v1','ce_uniform3',0),('confirm_p20261203_s20261211','fedlc_uniform3',0),('confirm_p20261203_s20261211','fedlc_uniform3',19)}
    input_hash_matches=0
    for run,r in runs.items():
        for path,expected in r['input_sha256_before_after_equal'].items():
            check('recorded_input_source_hash_matches',sha(path)==expected);input_hash_matches+=1
        summaries[run]={};client_metrics[run]={}
        for model,data in r['results'].items():
            rows=[]
            for c in range(20):
                source=read_npz(Path(r['input_root'])/f'{model}_client{c}.npz')
                output=read_npz(HERE/run/f'{model}_client{c}.npz')
                tr=source['train_logits'].astype(np.float64);ty=source['train_labels']
                raw=source.get('holdout_raw',source.get('raw')).astype(np.float64)
                labels=source.get('holdout_labels',source.get('labels'))
                learned=source.get('holdout_learned',source.get('learned')).astype(np.float64)
                old=data['clients'][c]
                check('train_holdout_ID_disjoint',not np.intersect1d(source['train_ids'],source['holdout_ids']).size)
                check('input_IDs_unique',len(np.unique(source['train_ids']))==len(ty) and len(np.unique(source['holdout_ids']))==len(labels))
                check('saved_output_IDs_unchanged',np.array_equal(source['train_ids'],output['train_ids']) and np.array_equal(source['holdout_ids'],output['holdout_ids']))
                counts=np.bincount(ty,minlength=10)
                prior=np.log((counts+1.)/(len(ty)+10)).astype(np.float32)
                check('prior_from_TRAIN_counts_only',np.array_equal(source['prior_shift'],prior))
                check('frozen_prioranchor_state_unchanged',np.array_equal(output['prioranchor_bias'],source['fitted_bias']))
                scores={'raw':raw,'analytic_prior':raw+prior,'prioranchor_bias':raw+source['fitted_bias'],
                        'learned_head':learned,'learned_head_plus_prior':learned+prior}
                for kind in KINDS:
                    state=output[kind];saved=old['optimization'][kind];theta=np.array(saved['parameters_float64'])
                    check('deployment_state_is_FP32',state.dtype==np.dtype('float32'))
                    check('deployment_dimension_and_bytes',state.shape==(DIMS[kind],) and state.nbytes==4*DIMS[kind])
                    check('FP32_state_is_saved_FP64_rounding',np.array_equal(state,theta.astype(np.float32)))
                    scores[kind]=mapping(raw,kind,state)
                    j,g,h=jgh(tr,ty,kind,theta);norm=float(np.max(np.abs(pg(theta,g,kind))))
                    maxkkt=max(maxkkt,norm);strict+=int(norm<=1e-8)
                    check('all_saved_KKT_within_resolution_tolerance',norm<1e-5)
                    close('saved_final_objective',j,saved['final_objective'],2e-7)
                    close('saved_projected_gradient',norm,saved['projected_gradient_inf'],2e-8)
                    eig=float(np.linalg.eigvalsh(h).min());check('all_saved_ridge_hessian_lower_bound',eig>=10-1e-7)
                    close('saved_hessian_min_eigenvalue',eig,saved['hessian_min_eigenvalue'],1e-7)
                    close('saved_anchor',saved['anchor'],anchor(kind),0.)
                    check('saved_fixed_ridge_and_dimension',saved['regularization']==10 and saved['dimension']==DIMS[kind])
                    b=bias_of(theta,kind)
                    if b is not None:check('saved_bias_is_centered',abs(b.mean())<1e-12)
                    atbound=kind in ('temperature_scaling','bcts') and theta[0]<=LOWER+1e-12
                    if kind in ('temperature_scaling','bcts'):check('saved_positive_scale_bound',theta[0]>=LOWER-1e-14 and state[0]>0)
                    if atbound:bound_count+=1;check('saved_bound_KKT_multiplier',g[0]>=-1e-8)
                    trace=saved['trace'];stop_counts[saved['stop_reason']]+=1
                    check('trace_objective_monotone',all(a['objective']>=b['objective']-1e-10 for a,b in zip(trace,trace[1:])))
                    check('trace_iteration_counts',len(trace)==saved['accepted_steps']+1 and saved['accepted_steps']<=50)
                    close('trace_initial_objective',trace[0]['objective'],jgh(tr,ty,kind,anchor(kind),False),2e-7)
                    close('trace_final_objective',trace[-1]['objective'],j,2e-7)
                    check('trace_step_fields',all(t['alpha']>0 and t['backtracks']<=40 for t in trace[1:]))
                    if saved['stop_reason']=='kkt_gradient_tolerance':check('strict_KKT_stop_reason_valid',norm<=1.1e-8)
                    else:check('resolution_stop_not_claimed_strict',saved['stop_reason']=='float64_objective_resolution')
                    state_audits.append({'run':run,'model':model,'client':c,'kind':kind,'objective':j,'projected_gradient_inf':norm,
                                         'hessian_min_eigenvalue':eig,'at_lower_bound':atbound,'stop_reason':saved['stop_reason'],'state_bytes':int(state.nbytes)})
                    if (run,model,c) in fixed_refit:
                        independent,fit=independent_fit(tr,ty,kind)
                        maxparam=float(np.max(np.abs(independent-theta)));gap=abs(fit['objective']-j)
                        check('independent_refit_parameter_agreement',maxparam<2e-5)
                        check('independent_refit_objective_agreement',gap<2e-7)
                        check('independent_refit_KKT',fit['projected_gradient_inf']<1e-5)
                        refits.append({'run':run,'model':model,'client':c,'kind':kind,'train_samples':len(ty),'parameter_max_error':maxparam,
                                       'objective_abs_gap':gap,'independent_parameters_float64':independent.tolist(),**fit})
                metrics={}
                for variant,score in scores.items():
                    check('saved_variant_logits_exact',np.array_equal(output[variant+'_holdout_logits'],score))
                    m=stats(score,labels);metrics[variant]=m
                    close('all_client_accuracy',m['accuracy'],old['metrics'][variant]['accuracy'],0.)
                    close('all_client_NLL',m['loss'],old['metrics'][variant]['loss'],1e-12)
                    check('all_client_sample_count',m['samples']==old['metrics'][variant]['samples'])
                # Strictly positive scalar temperature cannot change argmax.
                close('TS_preserves_raw_accuracy',metrics['temperature_scaling']['accuracy'],metrics['raw']['accuracy'],0.)
                rows.append({'client':c,'train_samples':len(ty),'holdout_samples':len(labels),'metrics':metrics})
            summaries[run][model]=summarize(rows);client_metrics[run][model]=rows
            for v,values in summaries[run][model].items():
                for key,value in values.items():close('all_summary_'+key,value,data['summaries'][v][key],1e-12)
    after={str(p):sha(p) for p in sorted(protected)}
    check('all_protected_inputs_unchanged',before==after)
    check('no_torch_or_scipy_imported','torch' not in sys.modules and 'scipy' not in sys.modules)
    check('no_forbidden_data_attempts',not FORBIDDEN)
    s=summaries['confirm_p20261203_s20261211']['fedlc_uniform3'];a=s['prioranchor_bias'];v=s['vector_scaling']
    check('confirm_vector_accuracy_better_but_NLL_worse',v['macro_accuracy']>a['macro_accuracy'] and v['macro_loss']>a['macro_loss'])
    report={'status':'PASS','scope':'NumPy-only local independent fixed-array audit; no official data/CNN/network/new ML training',
        'review_status':'LOCAL_INDEPENDENT_CHECK / external REVIEW_UNAVAILABLE',
        'check_count':sum(COUNTERS.values()),'checks_by_category':dict(COUNTERS),'maximum_errors':dict(ERRORS),
        'coverage':{'partitions':2,'frozen_models':7,'clients_per_model':20,'calibration_states':len(state_audits),
                    'variants_per_model':9,'client_variant_metric_rows':7*20*9,'summary_variant_rows':7*9,
                    'protected_files':len(before),'recorded_input_hashes_matched':input_hash_matches,
                    'independent_refit_client_cases':3,'independent_refits':len(refits)},
        'objective':'SUM of TRAIN CE + 10/2||theta-anchor||^2; never mean CE; same lambda does not equate dimensions/complexity, and effective mean-loss penalty differs with local sample count',
        'parameter_maps':{'zero_anchor_bias':'z+b; anchor b=0;10dim','temperature_scaling':'a*z; anchor a=1; a>=0.001;T=1/a;1dim',
            'bcts':'a*z+b; anchor(1,0); a>=0.001;11dim','vector_scaling':'A elementwise z+b; anchor(ones,0);A unrestricted;20dim'},
        'derivative_finite_difference':derivatives,'positive_scale_boundary_fixtures':boundary,
        'KKT':{'strict_threshold':1e-8,'states_with_recomputed_residual_le_strict':strict,'all_states':len(state_audits),
               'maximum_recomputed_projected_gradient_inf':maxkkt,'resolution_audit_threshold':1e-5,
               'stop_reason_counts':dict(stop_counts),'real_saved_scalar_bound_active_cases':bound_count,
               'note':'No saved state reaches a lower bound; two artificial balanced wrong-label fixtures explicitly test bound KKT. FP64 optimizer KKT is distinct from FP32 deployment rounding.'},
        'state_semantics':{'deployment_bytes':{'zero_anchor_bias':40,'temperature_scaling':4,'bcts':44,'vector_scaling':80,'prioranchor_bias':40},
            'rounding':'all 560 stored states equal float64 diagnostics rounded to float32 exactly',
            'scoring_arithmetic':'FP32 state parameters and frozen FP32 logits are promoted to FP64 for affine transform and stable NLL; not an executed all-FP32 neural deployment'},
        'independent_optimizer':'three preselected TRAIN-only cases, 12 refits; own reduced centered-bias QR basis and exact single-bound quadratic Newton step, different starting point; no source solver.fit called for these refits',
        'independent_refits':refits,'all_state_fit_audits':state_audits,'summaries':summaries,'client_metrics':client_metrics,
        'confirm_fedlc_tradeoff':{'prioranchor_bias':a,'vector_scaling':v,
            'vector_minus_anchor_macro_accuracy_pp':100*(v['macro_accuracy']-a['macro_accuracy']),
            'vector_minus_anchor_macro_NLL':v['macro_loss']-a['macro_loss'],
            'interpretation':'vector wins macro accuracy and q10 here but loses macro NLL; no universal calibration/accuracy dominance claim'},
        'reproduction_boundaries':['These are local independent ridge10 variants of established TS/BCTS/vector-scaling, not exact official unregularized/held-out calibrator reproductions.',
            'FedCal uses learned order-preserving MLP and scaler aggregation; neither is reproduced here.',
            'Positive scalar bound, centered biases, unrestricted real vector scales, and distinct dimensions are disclosed; same lambda is not equal complexity.',
            'Original probe source passes only TRAIN logits/labels to fit; holdout is scoring-only. This audit reproduces every metric but cannot prove future selection will avoid reuse of holdout.',
            'Prior-anchor states are reused byte-for-byte; no prior-anchor refitting, original-data ID binding, new CNN inference, calibration claim guarantee, or external review is established.'],
        'source_and_result_hashes_before':before,'source_and_result_hashes_after':after,
        'audit_source_sha256':sha(__file__),'runtime':{'numpy':np.__version__,'python':sys.executable,'seconds':time.perf_counter()-began},
        'forbidden_opens':FORBIDDEN}
    (HERE/'LOCAL_INDEPENDENT_CHECK.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
    text=(f"独立NumPy核查PASS：{sum(COUNTERS.values())}项检查；2组固定数组、7模型×20client，560个校准态、1260组client×variant指标及63组汇总全部复算。所有准确率精确一致，NLL/宏均值、holdout样本加权、client q10误差≤{max(ERRORS['all_client_NLL'],*[ERRORS['all_summary_'+k] for k in ('macro_accuracy','macro_loss','weighted_accuracy','weighted_loss','client_accuracy_q10')]):.3g}。\\n\\n"
      "映射核对：zero-anchor为z+b(10维)；TS为az、a≥.001(1维)；BCTS为az+b(11维)；VS为A⊙z+b(20维，A允许负值)。目标是ΣTRAIN CE＋10/2||θ−anchor||²，不是mean CE；bias中心化。四类解析grad/Hessian与独立公式、有限差分一致，ridge特征值≥10（舍入容差）。3个预先固定TRAIN案例×4类独立重拟合一致；不使用holdout GT拟合。\\n\\n"
      f"560态的FP32舍入与40/4/44/80B部署态完全核对；评分实际使用FP64仿射/NLL。轨迹单调，{stop_counts['kkt_gradient_tolerance']}次按KKT阈值结束，{stop_counts['float64_objective_resolution']}次按FP64分辨率结束，实际{strict}/560态满足1e−8，最大投影梯度{maxkkt:.3g}。保存态未触边界，另2个synthetic边界案例验证正a及KKT。\\n\\n"
      f"新分区FedLC uniform3：VS宏准确率{100*v['macro_accuracy']:.4f}%、CE {v['macro_loss']:.5f}；prior-anchor为{100*a['macro_accuracy']:.4f}%、{a['macro_loss']:.5f}。VS准确率高{100*(v['macro_accuracy']-a['macro_accuracy']):.4f}pp、CE反而高{v['macro_loss']-a['macro_loss']:.5f}；q10也由{100*a['client_accuracy_q10']:.4f}%到{100*v['client_accuracy_q10']:.4f}%。这是取舍，不能宣称prior-anchor全胜。\\n\\n"
      f"{len(before)}个源/输入/结果文件哈希前后不变；未读取原始数据/TEST、未加载CNN、无联网。官方复现差异：本地TRAIN拟合、ridge10、固定约束/中心化；没有FedCal的MLP、保序或聚合。同λ不同维度及样本量不代表等复杂度；holdout仅评分，无网格选型。LOCAL_INDEPENDENT_CHECK；外部REVIEW_UNAVAILABLE。")
    # Literal escapes above become paragraph separators here.
    text=text.replace('\\n','\n')
    assert len(text)<=1200,len(text)
    (HERE/'LOCAL_INDEPENDENT_CHECK.md').write_text(text+'\n')
    print(json.dumps({'status':'PASS','checks':sum(COUNTERS.values()),'states':len(state_audits),'metrics':1260,
                      'report_characters':len(text),'max_KKT':maxkkt,'confirm_fedlc_tradeoff':report['confirm_fedlc_tradeoff']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
