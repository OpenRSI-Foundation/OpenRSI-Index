"""Fixed aggregation and comparison, used by Judge and host state admission."""
import math
import statistics

COUNTS = {'en': 1088, 'zh': 2020}
WORKLOADS = tuple(f'{mode}/{lang}' for mode in ('nonstream', 'stream') for lang in COUNTS)
BASE_METRICS = ('qps', 'rtfx', 'latency_mean_s', 'latency_p95_s', 'rtf_mean',
                'rtf_p95', 'corpus_wer', 'coverage', 'warmup_coverage')
STREAM_METRICS = ('ttft_mean_s', 'ttft_p95_s', 'inter_chunk_mean_s',
                  'inter_chunk_p95_s', 'single_delta_fraction',
                  'empty_delta_count', 'last_delta_to_receipt_mean_s')


def finite(x, *, positive=False):
    return type(x) in (int, float) and math.isfinite(x) and (x > 0 if positive else x >= 0)


def aggregate(repeats):
    if set(repeats) != set(WORKLOADS):
        raise ValueError('incomplete workload coverage')
    means, ranges = {}, {}
    for key in WORKLOADS:
        rows = repeats[key]
        if not isinstance(rows, list) or len(rows) != 3:
            raise ValueError('three complete repeats required')
        names = BASE_METRICS + (STREAM_METRICS if key.startswith('stream/') else ())
        expected = COUNTS[key.split('/')[1]]
        for row in rows:
            if set(row) != set(names):
                raise ValueError('aggregate metric schema mismatch')
            if row['coverage'] != expected or row['warmup_coverage'] != expected:
                raise ValueError('request coverage mismatch')
        means[key], ranges[key] = {}, {}
        for name in names:
            values = [row[name] for row in rows]
            if name.startswith('inter_chunk') and any(x is None for x in values):
                # Three-run metrics exist only when all three repeats have intervals.
                if any(x is not None and not finite(x) for x in values):
                    raise ValueError('invalid optional interval metric')
                means[key][name] = None
                ranges[key][name] = None
                continue
            if not all(finite(x, positive=name in ('qps', 'rtfx')) for x in values):
                raise ValueError('nonfinite or negative metric')
            means[key][name] = statistics.mean(values)
            ranges[key][name] = [min(values), max(values)]
    return means, ranges


def compare(current, b0, bk):
    failed, ratios, cumulative = [], {}, {}
    for key in WORKLOADS:
        c, old, initial = current[key], bk[key], b0[key]
        ratios[key], cumulative[key] = {}, {}
        for metric in c:
            for target, base in ((ratios, old), (cumulative, initial)):
                a, b = c[metric], base[metric]
                target[key][metric] = a / b if a is not None and b not in (None, 0) else None
        if c['corpus_wer'] > 0.02:
            failed.append(key + ':corpus_wer>0.02')
        for metric in ('qps', 'rtfx'):
            if c[metric] < 0.97 * old[metric]:
                failed.append(key + ':' + metric + '<97% of Bk')
        latency = ['latency_mean_s', 'latency_p95_s', 'rtf_mean', 'rtf_p95']
        if key.startswith('stream/'):
            latency += ['ttft_mean_s', 'ttft_p95_s', 'inter_chunk_mean_s', 'inter_chunk_p95_s']
        for metric in latency:
            if c[metric] is not None and old[metric] is not None and c[metric] > 1.03 * old[metric]:
                failed.append(key + ':' + metric + '>103% of Bk')
    def modes(base):
        return {m: math.sqrt((current[m+'/en']['qps'] / base[m+'/en']['qps']) *
                             (current[m+'/zh']['qps'] / base[m+'/zh']['qps']))
                for m in ('stream', 'nonstream')}
    local, initial = modes(bk), modes(b0)
    for mode, ratio in local.items():
        if ratio <= 1:
            failed.append(mode + ':mode ratio must exceed 1 against Bk')
    reward = initial['stream'] ** (2/3) * initial['nonstream'] ** (1/3)
    if not finite(reward, positive=True):
        raise ValueError('reward must be positive and finite')
    return {'failed': failed, 'ratios_bk': ratios, 'ratios_b0': cumulative,
            'mode_ratios_bk': local, 'mode_ratios_b0': initial, 'reward': reward}
