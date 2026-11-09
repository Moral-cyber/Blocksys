"""Strict lazy final-TEST loaders, TRAIN-count assignment and input adapters.

Importing this module opens no files. Official loaders require a root-lock and
execute=True; parsers and assignment can be checked entirely in memory.
"""
from pathlib import Path
import gzip
import hashlib
import io
import json
import pickle
import platform
import struct
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
TASKS={'cifar':{'partition_seed':20261217,'seeds':[20261231,20261232,20261233]},
       'fashion':{'partition_seed':20261219,'seeds':[20261241,20261242,20261243]}}
OFFICIAL_MD5={'test_batch':'40351d587109b95175f43aff81a1287e',
    't10k-images-idx3-ubyte.gz':'bef4ecab320f06d8554ea6380940ec79',
    't10k-labels-idx1-ubyte.gz':'bb300cfdad3c16e7a12a480ee83cd310'}
ASSIGNMENT_SEED=20261301
CLIENTS=20
CLASSES=10
PER_CLASS=1000
TEST_ROWS=10000


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    return sha_bytes(Path(path).read_bytes())


def array_sha(value,dtype):
    return sha_bytes(np.ascontiguousarray(value,dtype=dtype).tobytes())


def canonical_sha(value):
    return sha_bytes(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())


def runtime():
    return {'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
        'assignment_rng':'numpy.random.Generator(PCG64(SeedSequence([20261301,class_id])))',
        'byte_order':'hash arrays explicitly little-endian; quota integer arithmetic'}


def validate_root_lock(root_lock,execute,task):
    """Read only the explicit lock/PLAN/code, never a TEST path here."""
    if execute is not True:
        raise ValueError('official TEST bytes require explicit --execute')
    if task not in TASKS or root_lock is None:
        raise ValueError('explicit task and root-lock required')
    lock_path=Path(root_lock).resolve()
    lock=json.loads(lock_path.read_text())
    if lock.get('status')!='ROOT_LOCKED_FINAL_TEST' or task not in lock.get('tasks',[]):
        raise ValueError('root-lock does not authorize this final TEST task')
    plan=HERE/'FINAL_TEST_PLAN.md'
    if lock.get('plan_sha256')!=file_sha(plan):
        raise ValueError('root-lock PLAN SHA mismatch')
    for name in ('data.py','scorer.py'):
        if lock.get('source_sha256',{}).get(name)!=file_sha(HERE/name):
            raise ValueError('root-lock helper SHA mismatch: '+name)
    return {'root_lock':str(lock_path),'root_lock_sha256':file_sha(lock_path),
        'plan_sha256':lock['plan_sha256'],'source_sha256':lock['source_sha256']}


def _verified_file(path,expected_md5):
    path=Path(path)
    payload=path.read_bytes()
    md5=hashlib.md5(payload).hexdigest()
    if md5!=expected_md5:
        raise ValueError('official TEST file MD5 mismatch: '+path.name)
    return payload,{'path':str(path.resolve()),'bytes':len(payload),'md5':md5,
        'expected_official_md5':expected_md5,'sha256':sha_bytes(payload),'official_md5_verified':True}


def _labels(labels):
    y=np.asarray(labels)
    if (y.shape!=(TEST_ROWS,) or y.dtype.kind not in 'iu' or np.any(y<0) or np.any(y>=CLASSES)):
        raise ValueError('exact10000 integer official TEST labels0..9 required')
    y=y.astype(np.int64,copy=True)
    if not np.array_equal(np.bincount(y,minlength=CLASSES),np.full(CLASSES,PER_CLASS)):
        raise ValueError('official TEST must have exactly1000 rows in every class')
    return y


def _latin1_encode(value,encoding,errors='strict'):
    if not isinstance(value,str) or encoding not in ('latin1','latin-1') or errors!='strict':
        raise pickle.UnpicklingError('only exact NumPy pickle latin1 bytes reconstruction permitted')
    return value.encode('latin1')


class _CifarUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if module in ('numpy.core.multiarray','numpy._core.multiarray') and name=='_reconstruct':
            return np.core.multiarray._reconstruct
        if module=='numpy' and name in ('ndarray','dtype'):
            return getattr(np,name)
        if module=='_codecs' and name=='encode':
            return _latin1_encode
        raise pickle.UnpicklingError('CIFAR pickle global not allowed: '+module+'.'+name)


def parse_cifar_bytes(payload):
    if not isinstance(payload,bytes) or not 100<len(payload)<40_000_000:
        raise ValueError('bounded official CIFAR TEST byte payload required')
    stream=io.BytesIO(payload)
    obj=_CifarUnpickler(stream,encoding='latin1').load()
    if stream.read(1) or not isinstance(obj,dict):
        raise ValueError('one exact CIFAR pickle dictionary required')
    get=lambda key:obj[key] if key in obj else obj[key.encode()]
    pixels=get('data')
    if not isinstance(pixels,np.ndarray) or pixels.dtype!=np.uint8 or pixels.shape!=(TEST_ROWS,3072):
        raise ValueError('official CIFAR TEST uint8(10000,3072) required')
    labels=_labels(get('labels'))
    images=pixels.reshape(TEST_ROWS,3,32,32).transpose(0,2,3,1).copy()
    return images,labels,{'decoded_image_shape':list(images.shape),
        'label_source':'official test_batch dictionary labels in original row order',
        'labels_int64le_sha256':array_sha(labels,'<i8'),
        'decoded_images_uint8_sha256':array_sha(images,'u1')}


def parse_idx_bytes(payload,magic,shape):
    header_size=4*(1+len(shape))
    if not isinstance(payload,bytes) or len(payload)!=header_size+int(np.prod(shape)):
        raise ValueError('exact IDX header/payload byte count required')
    header=struct.unpack('>'+'I'*(len(shape)+1),payload[:header_size])
    if header!=(magic,*shape):
        raise ValueError('IDX magic/shape mismatch')
    return np.frombuffer(payload,dtype=np.uint8,offset=header_size).reshape(shape).copy()


def parse_fashion_gzip_bytes(image_gzip,label_gzip):
    def bounded_decode(payload,expected):
        with gzip.GzipFile(fileobj=io.BytesIO(payload),mode='rb') as handle:
            result=handle.read(expected+1)
        if len(result)!=expected:
            raise ValueError('Fashion gzip decoded byte count mismatch')
        return result
    image_idx=bounded_decode(image_gzip,16+TEST_ROWS*28*28)
    label_idx=bounded_decode(label_gzip,8+TEST_ROWS)
    images=parse_idx_bytes(image_idx,2051,(TEST_ROWS,28,28))
    labels=_labels(parse_idx_bytes(label_idx,2049,(TEST_ROWS,)))
    return images,labels,{'decoded_image_shape':list(images.shape),
        'label_source':'official t10k-labels IDX payload in original row order',
        'labels_int64le_sha256':array_sha(labels,'<i8'),
        'label_idx_decompressed_sha256':sha_bytes(label_idx),
        'label_uint8_payload_sha256':sha_bytes(label_idx[8:]),
        'image_idx_decompressed_sha256':sha_bytes(image_idx),
        'decoded_images_uint8_sha256':array_sha(images,'u1')}


def load_cifar_test(raw,*,root_lock=None,execute=False):
    lock=validate_root_lock(root_lock,execute,'cifar')
    path=Path(raw)/'cifar-10-batches-py/test_batch'
    payload,identity=_verified_file(path,OFFICIAL_MD5['test_batch'])
    images,labels,decoded=parse_cifar_bytes(payload)
    return images,labels,{'task':'cifar','official_split':'official TEST only',
        'files':[identity],**decoded,'authorization':lock,'runtime':runtime()}


def load_fashion_test(raw,*,root_lock=None,execute=False):
    lock=validate_root_lock(root_lock,execute,'fashion')
    image_bytes,images_identity=_verified_file(Path(raw)/'t10k-images-idx3-ubyte.gz',
        OFFICIAL_MD5['t10k-images-idx3-ubyte.gz'])
    label_bytes,labels_identity=_verified_file(Path(raw)/'t10k-labels-idx1-ubyte.gz',
        OFFICIAL_MD5['t10k-labels-idx1-ubyte.gz'])
    images,labels,decoded=parse_fashion_gzip_bytes(image_bytes,label_bytes)
    return images,labels,{'task':'fashion','official_split':'official TEST only',
        'files':[images_identity,labels_identity],**decoded,'authorization':lock,'runtime':runtime()}


def validate_train_counts(counts):
    counts=np.asarray(counts)
    if (counts.shape!=(CLIENTS,CLASSES) or counts.dtype.kind not in 'iu'
            or np.any(counts<0) or np.any(counts>1_000_000)):
        raise ValueError('20x10 bounded nonnegative integer local TRAIN counts required')
    if np.any(counts.sum(axis=1)==0):
        raise ValueError('every TRAIN client must be nonempty')
    return counts.astype(np.int64,copy=True)


def largest_remainder(train_counts,per_class=PER_CLASS):
    counts=validate_train_counts(train_counts)
    if not isinstance(per_class,int) or not 1<=per_class<=10000:
        raise ValueError('bounded positive integer class rows required')
    weights=counts+1
    denominators=weights.sum(axis=0)
    numerators=weights*per_class
    quota=numerators//denominators
    remainders=numerators%denominators
    for label in range(CLASSES):
        left=per_class-int(quota[:,label].sum())
        order=sorted(range(CLIENTS),key=lambda client:(-int(remainders[client,label]),client))
        quota[order[:left],label]+=1
    assert np.array_equal(quota.sum(axis=0),np.full(CLASSES,per_class))
    assert np.all(np.abs(quota*denominators-numerators)<denominators)
    return quota,weights,denominators


def assignment_digest(groups):
    h=hashlib.sha256()
    for group in groups:
        h.update(struct.pack('<I',len(group)))
        h.update(np.asarray(group,dtype='<u4').tobytes())
    return h.hexdigest()


def assign_clients(labels,train_counts,*,task,train_manifest_sha256):
    if task not in TASKS or not isinstance(train_manifest_sha256,str) or len(train_manifest_sha256)!=64:
        raise ValueError('declared task and frozen TRAIN manifest SHA required')
    y=_labels(labels)
    counts=validate_train_counts(train_counts)
    quotas,weights,denominators=largest_remainder(counts)
    groups=[[] for _ in range(CLIENTS)]
    owner=np.full(TEST_ROWS,-1,dtype=np.int64)
    for label in range(CLASSES):
        ids=np.flatnonzero(y==label)
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([ASSIGNMENT_SEED,label])))
        rng.shuffle(ids)
        start=0
        for client in range(CLIENTS):
            end=start+int(quotas[client,label])
            selected=ids[start:end]
            groups[client].extend(selected.tolist())
            owner[selected]=client
            start=end
    groups=[np.asarray(group,dtype=np.uint32) for group in groups]
    if any(not len(group) for group in groups):
        raise ValueError('defined TEST protocol produced empty client; no resampling permitted')
    flat=np.concatenate(groups)
    if len(flat)!=TEST_ROWS or not np.array_equal(np.sort(flat),np.arange(TEST_ROWS)) or np.any(owner<0):
        raise AssertionError('TEST rows not assigned exactly once')
    actual=np.array([np.bincount(y[group],minlength=CLASSES) for group in groups])
    if not np.array_equal(actual,quotas):
        raise AssertionError('realized TEST quota mismatch')
    train_prior=counts/counts.sum(axis=1,keepdims=True)
    test_prior=quotas/quotas.sum(axis=1,keepdims=True)
    difference=test_prior-train_prior
    assignment={'schema':'train_count_defined_label_skew_test_v1','task':task,**TASKS[task],
        'scope':'newly defined label-skew TEST protocol; not an official benchmark client partition',
        'train_manifest_sha256':train_manifest_sha256,
        'train_counts':counts.tolist(),'train_counts_int64le_sha256':array_sha(counts,'<i8'),
        'labels_int64le_sha256':array_sha(y,'<i8'),'assignment_seed':ASSIGNMENT_SEED,
        'class_rng':'PCG64(SeedSequence([20261301,class_id]))','tie_rule':'descending exact integer remainder then ascending client ID',
        'smoothed_client_given_class':(weights/denominators).tolist(),
        'test_class_quotas':quotas.tolist(),'client_sizes':[len(ids) for ids in groups],
        'client_ids':[ids.tolist() for ids in groups],
        'assignment_hash':assignment_digest(groups),'owner_int64le_sha256':array_sha(owner,'<i8'),
        'train_class_given_client':train_prior.tolist(),'test_class_given_client':test_prior.tolist(),
        'test_minus_train_class_frequency':difference.tolist(),
        'per_client_total_variation':(.5*np.abs(difference).sum(axis=1)).tolist(),
        'per_client_max_abs_class_frequency_difference':np.abs(difference).max(axis=1).tolist(),
        'class_frequency_scope':'TRAIN weights determine P(client|class); balanced TEST class totals and integer quotas can change P(class|client); not exact TRAIN prior matching',
        'coverage_exactly_once':True,'runtime':runtime()}
    assignment['manifest_hash']=canonical_sha(assignment)
    return groups,assignment


def normalize_batch(images,task):
    pixels=np.asarray(images)
    if pixels.dtype!=np.uint8:
        raise ValueError('uint8 source pixels required')
    if task=='cifar' and pixels.ndim==4 and pixels.shape[1:]==(32,32,3):
        chw=pixels.transpose(0,3,1,2).copy()
    elif task=='fashion' and pixels.ndim==3 and pixels.shape[1:]==(28,28):
        padded=np.pad(pixels,((0,0),(2,2),(2,2)),constant_values=0)
        chw=np.repeat(padded[:,None],3,axis=1)
    else:
        raise ValueError('exact CIFAR HWC32 or Fashion gray28 batch required')
    result=chw.astype(np.float32)
    result/=np.float32(255.)
    result-=np.float32(.5)
    result/=np.float32(.5)
    return result
