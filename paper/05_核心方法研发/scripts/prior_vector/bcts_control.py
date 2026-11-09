"""11-dimensional prior-anchored BCTS control; frozen numerical code reused."""
import types
import numpy as np
import solver as s

SOLVER_SHA256='9c3196eb3e6ff7f5f32073e3bc4aae8d986ac7e44bdf0d719258343847d4dbae'
if s.sha(s.__file__)!=SOLVER_SHA256:
    raise ValueError('frozen main vector wrapper changed')


class Problem(s.base.Problem):
    def __init__(self,logits,labels,bias_anchor):
        super().__init__(logits,labels,'bcts')
        bias_anchor=np.asarray(bias_anchor,dtype=np.float64)
        if bias_anchor.shape!=(10,) or not np.isfinite(bias_anchor).all():
            raise ValueError('finite ten-dimensional bias anchor required')
        self.anchor[1:]=bias_anchor-bias_anchor.mean()


def fit(logits,labels,bias_anchor):
    """Bind the original BCTS fit code to a private configurable factory.

    No frozen globals are patched. Its a>=.001 constraint, KKT active set,
    Newton/Armijo schedule and stopping code are executed byte-for-byte.
    """
    bias_anchor=np.asarray(bias_anchor,dtype=np.float64).copy()
    def factory(z,y,kind):
        if kind!='bcts':
            raise ValueError('BCTS control cannot change calibration family')
        return Problem(z,y,bias_anchor)
    namespace=dict(s.base.fit.__globals__)
    namespace['Problem']=factory
    bound=types.FunctionType(s.base.fit.__code__,namespace,s.base.fit.__name__,
        s.base.fit.__defaults__,s.base.fit.__closure__)
    bound.__kwdefaults__=s.base.fit.__kwdefaults__
    state,diagnostic=bound(logits,labels,'bcts')
    diagnostic.update({'kind':'prioranchored_bcts_control',
        'role':'mechanism control only; not a new candidate or gate',
        'optimizer_binding':'exact frozen base.fit code with private Problem factory; original globals unchanged',
        'regularization_scale':10.,'regularization_bias':10.,
        'deployment_state':'one FP32 positive scalar a plus ten FP32 biases; 44 bytes',
        'design_dimension_caveat':'same ridge coefficient; one scale vs ten scales changes dimensionality and total ridge penalty'})
    return state,diagnostic


def transform(logits,state):
    return s.base.transform(logits,'bcts',state)
