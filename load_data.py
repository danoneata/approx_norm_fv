from collections import defaultdict
from collections import namedtuple
import cPickle
import functools
from itertools import izip
from itertools import product
import os
import pdb
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from ipdb import set_trace

from sklearn.preprocessing import Scaler
from yael import yael

from fisher_vectors.model.fv_model import FVModel
from fisher_vectors.model.sfv_model import SFVModel

from fisher_vectors.model.utils import power_normalize
from fisher_vectors.model.utils import L2_normalize
from fisher_vectors.model.utils import sstats_to_sqrt_features

SliceData = namedtuple('SliceData', ['fisher_vectors', 'counts', 'nr_descriptors'])


CACHE_PATH = '/scratch2/clear/oneata/tmp/joblib/'

hmdb_stab_dict = {
    'hmdb_split%d.stab' % ii :{
        'dataset_name': 'hmdb_split%d' % ii,
        'dataset_params': {
            'ip_type': 'dense5.track15mbh',
            'nr_clusters': 256,
            'suffix': '.per_slice.delta_15.stab.fold_%d' % ii,
        },
        'samples_chunk': 100,
        'eval_name': 'hmdb',
        'eval_params': {
        },
        'metric': 'accuracy',
    } for ii in xrange(1, 4)}


hmdb_all_descs_dict = {
    'hmdb_split%d.delta_5.all_descs' % ii :{
        'dataset_name': 'hmdb_split%d' % ii,
        'dataset_params': {
            'ip_type': 'dense5.track15hog,hof,mbh',
            'nr_clusters': 256,
            'suffix': '.per_slice.delta_5.fold_%d' % ii,
            'separate_pca': True,
        },
        'samples_chunk': 100,
        'eval_name': 'hmdb',
        'eval_params': {
        },
        'metric': 'accuracy',
    } for ii in xrange(1, 4)}


cache_dir = os.path.expanduser('~/scratch2/tmp')
CFG = {
    'trecvid11_devt': {
        'dataset_name': 'trecvid12',
        'dataset_params': {
            'ip_type': 'dense5.track15mbh',
            'nr_clusters': 256,
            'suffix': '.per_slice.small.delta_60.skip_1',
        },
        'eval_name': 'trecvid12',
        'eval_params': {
            'split': 'devt',
        },
        'metric': 'average_precision',
    },
    'hollywood2':{
        'dataset_name': 'hollywood2',
        'dataset_params': {
            'ip_type': 'dense5.track15mbh',
            'nr_clusters': 256,
            'suffix': '.per_slice.delta_60',
        },
        'samples_chunk': 25,
        'eval_name': 'hollywood2',
        'eval_params': {
        },
        'metric': 'average_precision',
    },
    'hollywood2.delta_5.small':{
        'dataset_name': 'hollywood2',
        'dataset_params': {
            'ip_type': 'dense5.track15mbh',
            'nr_clusters': 50,
            'suffix': '.per_slice.delta_5',
            'tmp_suffix': '_spm131',
        },
        'samples_chunk': 25,
        'eval_name': 'hollywood2',
        'spms': [(1, 1, 1), (1, 1, 2), (1, 3, 1)],
        'encodings': ['fv', 'sfv'],
        'eval_params': {
        },
        'metric': 'average_precision',
    },
    'hollywood2.delta_5':{
        'dataset_name': 'hollywood2',
        'dataset_params': {
            'ip_type': 'dense5.track15mbh',
            'nr_clusters': 1000,
            'suffix': '.per_slice.delta_5',
            'tmp_suffix': '_spm131',
        },
        'samples_chunk': 25,
        'eval_name': 'hollywood2',
        'spms': [(1, 1, 1), (1, 1, 2), (1, 3, 1)],
        'encodings': ['fv', 'sfv'],
        'eval_params': {
        },
        'metric': 'average_precision',
    },
    'hollywood2.delta_30':{
        'dataset_name': 'hollywood2',
        'dataset_params': {
            'ip_type': 'dense5.track15mbh',
            'nr_clusters': 256,
            'suffix': '.per_slice.delta_30',
        },
        'samples_chunk': 25,
        'eval_name': 'hollywood2',
        'eval_params': {
        },
        'metric': 'average_precision',
    },
    'hollywood2.delta_5.all_descs':{
        'dataset_name': 'hollywood2',
        'dataset_params': {
            'ip_type': 'dense5.track15hog,hof,mbh',
            'nr_clusters': 256,
            'suffix': '.delta_5',
            'separate_pca': True,
        },
        'samples_chunk': 25,
        'eval_name': 'hollywood2',
        'eval_params': {
        },
        'metric': 'average_precision',
    },
    'hmdb.delta_5':{
        'dataset_name': 'hmdb',
        'dataset_params': {
            'ip_type': 'dense5.track15mbh',
            'nr_clusters': 1000,
            'suffix': '.per_slice.delta_5',
            'tmp_suffix': '_spm131',
        },
        'samples_chunk': 100,
        'eval_name': 'hmdb',
        'spms': [(1, 1, 1), (1, 1, 2), (1, 3, 1)],
        'encodings': ['fv', 'sfv'],
        'eval_params': {
        },
        'metric': 'accuracy',
    },
    'hmdb_split1':{
        'dataset_name': 'hmdb_split1',
        'dataset_params': {
            'ip_type': 'dense5.track15mbh',
            'nr_clusters': 256,
            'suffix': '.per_slice.delta_30',
        },
        'samples_chunk': 100,
        'eval_name': 'hmdb',
        'eval_params': {
        },
        'metric': 'accuracy',
    },
    'cc':{
        'dataset_name': 'cc',
        'dataset_params': {
            'ip_type': 'dense5.track15mbh',
            'nr_clusters': 128,
            'suffix': '',
        },
        'eval_name': 'cc',
        'eval_params': {
        },
        'metric': 'average_precision',
        'chunk_size': 30,
    },
    'cc.no_stab':{
        'dataset_name': 'cc',
        'dataset_params': {
            'ip_type': 'dense5.track15mbh',
            'nr_clusters': 128,
            'suffix': '.delta_5.no_stab',
        },
        'eval_name': 'cc',
        'eval_params': {
        },
        'metric': 'average_precision',
        'chunk_size': 5,
    },
    'cc.stab':{
        'dataset_name': 'cc',
        'dataset_params': {
            'ip_type': 'dense5.track15mbh',
            'nr_clusters': 128,
            'suffix': '.stab',
        },
        'eval_name': 'cc',
        'eval_params': {
        },
        'metric': 'average_precision',
        'chunk_size': 1,
    },
    'cc.delta_5.all_descs':{
        'dataset_name': 'cc',
        'dataset_params': {
            'ip_type': 'dense5.track15hog,hof,mbh',
            'nr_clusters': 256,
            'suffix': '.delta_5.separate_pca',
            'separate_pca': True,
        },
        'eval_name': 'cc',
        'eval_params': {
        },
        'metric': 'average_precision',
        'chunk_size': 5,
    },
    'cc.delta_5.all_descs.combined_pca':{
        'dataset_name': 'cc',
        'dataset_params': {
            'ip_type': 'dense5.track15hog,hof,mbh',
            'nr_clusters': 256,
            'suffix': '.delta_5.combined_pca',
            'separate_pca': False,
            'nr_pca_dims': 192,
        },
        'eval_name': 'cc',
        'eval_params': {
        },
        'metric': 'average_precision',
        'chunk_size': 5,
    },
    'duch09':{
        'dataset_name': 'duch09',
        'dataset_params': {
            'ip_type': 'dense5.track15mbh',
            'nr_clusters': 128,
            'suffix': '',
        },
        'eval_name': 'cc',
        'eval_params': {
        },
        'metric': 'average_precision',
        'chunk_size': 30,
    },
    'duch09.delta_5.all_descs':{
        'dataset_name': 'duch09',
        'dataset_params': {
            'ip_type': 'dense5.track15hog,hof,mbh',
            'nr_clusters': 256,
            'suffix': '.delta_5.separate_pca',
            'separate_pca': True,
        },
        'eval_name': 'cc',
        'eval_params': {
        },
        'metric': 'average_precision',
        'chunk_size': 5,
    },
}

CFG.update(hmdb_stab_dict)
CFG.update(hmdb_all_descs_dict)


def my_cacher(*args):

    def loader(file, format):
        if format in ('cp', 'cPickle'):
            result = cPickle.load(file)
        elif format in ('np', 'numpy'):
            result = np.load(file)
        else:
            assert False
        return result

    def dumper(file, result, format):
        if format in ('cp', 'cPickle'):
            cPickle.dump(result, file)
        elif format in ('np', 'numpy'):
            np.save(file, result)
        else:
            assert False

    store_format = args

    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            outfile = kwargs.get('outfile', tempfile.mkstemp()[1])
            if os.path.exists(outfile):
                with open(outfile, 'r') as ff:
                    return [loader(ff, sf) for sf in store_format]
                    # FIXME For compatibility reasons I used the old way --- ugly!
                    #if len(store_format) > 1:
                    #    return [loader(ff, sf) for sf in store_format]
                    #else:
                    #    return loader(ff, store_format[0])
            else:
                result = func(*args, **kwargs)
                with open(outfile, 'w') as ff:
                    for rr, sf in izip(result, store_format):
                        dumper(ff, rr, sf)
                    #if len(store_format) > 1:
                    #    for rr, sf in izip(result, store_format):
                    #        dumper(ff, rr, sf)
                    #else:
                    #    dumper(ff, result, store_format[0])
                return result
        return wrapped

    return decorator


@my_cacher('np', 'np', 'cp')
def load_video_data(
    dataset, samples, verbose=0, outfile=None, analytical_fim=True,
    pi_derivatives=False, sqrt_nr_descs=False, spm=None, encoding='fv'):

    jj = 0
    N = len(samples)
    D, K = dataset.D, dataset.VOC_SIZE
    FV_DIM = 2 * K * D if encoding == 'fv' else 2 * 3 * K
    N_BINS = np.prod(spm)

    tr_video_data = np.zeros((N, N_BINS * FV_DIM), dtype=np.float32)
    tr_video_counts = np.zeros((N, N_BINS * K), dtype=np.float32)
    tr_video_labels = []
    tr_video_names = []

    def prepare_binned_data(X, C, nn):
        nn = nn.T
        X = X.reshape(nn.shape[0], nn.shape[1], FV_DIM)
        C = C.reshape(nn.shape[0], nn.shape[1], K)
        return X, C, nn

    def aggregate_1(X, C, nn):
        nn = nn[nn != 0][:, np.newaxis]
        Xagg = (X * nn).sum(axis=0) / nn.sum()
        Cagg = (C * nn).sum(axis=0) / nn.sum()
        return Xagg, Cagg

    # Treat differently the data stored from spatial pyramids.
    def aggregate_spm_1(X, C, nn):
        X, C, nn = prepare_binned_data(X, C, nn)
        Xagg = (X * nn).sum(axis=(0, 1)) / nn.sum(axis=(0, 1))
        Cagg = (C * nn).sum(axis=(0, 1)) / nn.sum(axis=(0, 1))
        return Xagg, Cagg

    def aggregate_spm_h3(X, C, nn):
        X, C, nn = prepare_binned_data(X, C, nn)
        Xagg = (X * nn).sum(axis=0) / nn.sum(axis=0)
        Cagg = (C * nn).sum(axis=0) / nn.sum(axis=0)
        return Xagg.flatten(), Cagg.flatten()

    def aggregate_spm_t2(X, C, nn):
        X, C, nn = prepare_binned_data(X, C, nn)
        NS = X.shape[0]  # Number of slices.
        Xagg = np.vstack([
            (X * nn)[: NS / 2].sum(axis=(0, 1)) / nn[: NS / 2].sum(axis=(0, 1)),
            (X * nn)[NS / 2 :].sum(axis=(0, 1)) / nn[NS / 2 :].sum(axis=(0, 1))])
        Cagg = np.vstack([
            (C * nn)[: NS / 2].sum(axis=(0, 1)) / nn[: NS / 2].sum(axis=(0, 1)),
            (C * nn)[NS / 2 :].sum(axis=(0, 1)) / nn[NS / 2 :].sum(axis=(0, 1))])
        return Xagg.flatten(), Cagg.flatten()

    AGG = {
        None: aggregate_1,
        (1, -1, -1): aggregate_1, # FIXME Hack.
        (1, 1, 1): aggregate_spm_1,
        (1, 1, 2): aggregate_spm_t2,
        (1, 3, 1): aggregate_spm_h3,
    }
    aggregate = AGG[spm]

    for sample in samples:

        fv, ii, cc, _ = load_sample_data(
            dataset, sample, return_info=True, analytical_fim=analytical_fim,
            encoding=encoding)

        if len(fv) == 0 or str(sample) in tr_video_names:
            continue

        nd = ii['nr_descs']
        ll = ii['label']

        fv_agg, cc_agg = aggregate(fv, cc, nd)

        tr_video_data[jj] = fv_agg
        tr_video_counts[jj] = cc_agg

        tr_video_labels.append(ll)
        tr_video_names.append(str(sample))

        jj += 1

        if verbose:
            print '%5d %5d %s' % (jj, fv.shape[0], sample.movie)

    tr_video_data[np.isnan(tr_video_data)] = 0
    tr_video_counts[np.isnan(tr_video_counts)] = 0

    return tr_video_data[:jj], tr_video_counts[:jj], tr_video_labels[:jj]


def load_sample_data(
    dataset, sample, analytical_fim=False, pi_derivatives=False,
    sqrt_nr_descs=False, return_info=False, encoding='fv'):

    ENC_PARAMS = {
        'fv': {
            'suffix_enc': '',
            'get_dim': lambda gmm: gmm.k * (2 * gmm.d + 1),
            'sstats_to_features': FVModel.sstats_to_features,
            'sstats_to_normalized_features': FVModel.sstats_to_normalized_features,
        },
        'sfv': {
            'suffix_enc': '_sfv',
            'get_dim': lambda gmm: gmm.k * (2 * 3 + 1),
            'sstats_to_features': SFVModel.spatial_sstats_to_spatial_features,
        },
    }

    if str(sample) in ('train', 'test'):
        stats_file = "%s.dat" % sample
        labels_file = "labels_%s.info" % sample
        info_file = "info_%s.info" % sample
    else:
        stats_tmp = "stats.tmp%s%s/%s" % (
            dataset.SUFFIX_STATS, ENC_PARAMS[encoding]['suffix_enc'], sample)

        stats_file = "%s.dat" % stats_tmp
        labels_file = "%s.info" % stats_tmp
        info_file = "%s.info" % stats_tmp

    stats_path = os.path.join(dataset.SSTATS_DIR, stats_file)
    labels_path = os.path.join(dataset.SSTATS_DIR, labels_file)
    info_path = os.path.join(dataset.SSTATS_DIR, info_file)

    with open(dataset.GMM, 'r') as ff:
        gmm = yael.gmm_read(ff)

    K = gmm.k
    D = ENC_PARAMS[encoding]['get_dim'](gmm)

    data = np.fromfile(stats_path, dtype=np.float32)
    data = data.reshape(-1, D)
    counts = data[:, : K]

    if analytical_fim:
        data = ENC_PARAMS[encoding]['sstats_to_normalized_features'](data, gmm)
    else:
        data = ENC_PARAMS[encoding]['sstats_to_features'](data, gmm)

    with open(labels_path, 'r') as ff:
        labels = cPickle.load(ff)

    with open(info_path, 'r') as ff:
        info = cPickle.load(ff)

    if sqrt_nr_descs:
        T = info['nr_descs']
        T = np.sqrt(T)[:, np.newaxis]
    else:
        T = 1.

    if pi_derivatives or encoding == 'sfv':
        # For the spatial Fisher vector encoding I drop the `pi_derivatives`,
        # so I need the full vector, no matter the value of the `pi_derivates`
        # parameter.
        idxs = slice(D)
    else:
        idxs = slice(K, D)

    if return_info:
        return T * data[:, idxs], labels, counts, info
    else:
        return T * data[:, idxs], labels, counts


def load_kernels(
    dataset, tr_norms=['std', 'sqrt', 'L2'], te_norms=['std', 'sqrt', 'L2'],
    analytical_fim=False, pi_derivatives=False, sqrt_nr_descs=False,
    only_train=False, verbose=0, do_plot=False, outfile=None):

    tr_outfile = outfile % "train" if outfile is not None else outfile

    # Load sufficient statistics.
    samples, _ = dataset.get_data('train')
    tr_data, tr_counts, tr_labels = load_video_data(
        dataset, samples, outfile=tr_outfile, analytical_fim=analytical_fim,
        pi_derivatives=pi_derivatives, sqrt_nr_descs=sqrt_nr_descs, verbose=verbose)

    if verbose > 0:
        print "Train data: %dx%d" % tr_data.shape

    if do_plot:
        plot_fisher_vector(tr_data[0], 'before')

    scalers = []
    for norm in tr_norms:
        if norm == 'std':
            scaler = Scaler()
            tr_data = scaler.fit_transform(tr_data)
            scalers.append(scaler)
        elif norm == 'sqrt':
            tr_data = power_normalize(tr_data, 0.5)
        elif norm == 'sqrt_cnt':
            tr_data = approximate_signed_sqrt(
                tr_data, tr_counts, pi_derivatives=pi_derivatives)
        elif norm == 'L2':
            tr_data = L2_normalize(tr_data)
        if do_plot:
            plot_fisher_vector(tr_data[0], 'after_%s' % norm)

    tr_kernel = np.dot(tr_data, tr_data.T)

    if only_train:
        return tr_kernel, tr_labels, scalers, tr_data

    te_outfile = outfile % "test" if outfile is not None else outfile

    # Load sufficient statistics.
    samples, _ = dataset.get_data('test')
    te_data, te_counts, te_labels = load_video_data(
        dataset, samples, outfile=te_outfile, analytical_fim=analytical_fim,
        pi_derivatives=pi_derivatives, sqrt_nr_descs=sqrt_nr_descs, verbose=verbose)

    if verbose > 0:
        print "Test data: %dx%d" % te_data.shape

    ii = 0
    for norm in te_norms:
        if norm == 'std':
            te_data = scalers[ii].transform(te_data)
            ii += 1
        elif norm == 'sqrt':
            te_data = power_normalize(te_data, 0.5)
        elif norm == 'sqrt_cnt':
            te_data = approximate_signed_sqrt(
                te_data, te_counts, pi_derivatives=pi_derivatives)
        elif norm == 'L2':
            te_data = L2_normalize(te_data)

    te_kernel = np.dot(te_data, tr_data.T)

    return tr_kernel, tr_labels, te_kernel, te_labels



def approximate_signed_sqrt(data, counts, pi_derivatives=False, verbose=0):
    Nc, K = counts.shape
    Nd, dim = data.shape
    D = (dim / K - (1 if pi_derivatives else 0)) / 2
    assert Nc == Nd, 'Data and counts sizes do not correspond.'

    sqrtQ = np.sqrt(np.abs(counts))
    sqrtQ = np.hstack((
        sqrtQ if pi_derivatives else np.empty((Nd, 0)),
        sqrtQ.repeat(D, axis=1),
        sqrtQ.repeat(D, axis=1)))
    data = data / sqrtQ

    if verbose > 1:
        print '\tSquare rooting the counts'
        print '\t\tNumber of infinite values', data[np.isinf(data)].size
        print '\t\tNumber of NaN values', data[np.isnan(data)].size

    # Remove degenerated values.
    data[np.isnan(data) | np.isinf(data)] = 0.

    return data


def plot_fisher_vector(xx, name='oo'):
    ii = np.argmax(np.abs(xx))
    print "\tPloting Fisher vector."
    print "\t\t%30s -- maximum at %7d is %+.3f." % (name, ii, xx[ii])
    D = xx.size
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.vlines(np.arange(D), np.zeros(D), xx)
    plt.xlabel('$d$')
    plt.ylabel(r'$\nabla_{\theta}\mathbf{x}$')
    plt.savefig('/tmp/%s.png' % name)
