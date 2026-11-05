"""Joint-objective contract regressions; synthetic records, no GPU or task runs."""
import copy
import unittest
from unittest.mock import patch
from test_model_lane import ROOT, calibration, common, engine, noise, runs


class DualModeTests(unittest.TestCase):
    def setUp(self):
        self.nonstream = common.PROTOCOL['lanes'][0]['name']
        self.stream = self.nonstream + '_stream'
        self.initial = common.means(runs())

    def candidate(self, normal, stream):
        result = copy.deepcopy(self.initial)
        result[self.nonstream]['throughput_qps'] *= normal
        result[self.stream]['throughput_qps'] *= stream
        return result

    def accepted(self, values):
        return all(all(g.values()) for g in noise.compare(values, {'means': self.initial})[1].values())

    def test_both_modes_must_improve_even_when_weighted_reward_increases(self):
        for normal, stream in [(1, 1), (1.2, 1), (1, 1.2), (1.3, .99), (.99, 1.3), (.99, .99)]:
            with self.subTest(nonstream=normal, stream=stream):
                self.assertFalse(self.accepted(self.candidate(normal, stream)))
        self.assertTrue(self.accepted(self.candidate(1.01, 1.01)))
        self.assertGreater(noise.objective_reward(self.candidate(.99, 1.3), self.initial), 1)
        self.assertFalse(self.accepted(self.candidate(.99, 1.3)))

    def test_weighting_is_two_thirds_streaming_and_one_third_nonstreaming(self):
        values = self.candidate(1.1, 1.2)
        self.assertAlmostEqual(noise.objective_reward(values, self.initial), 1.2**(2/3)*1.1**(1/3))
        self.assertGreater(noise.objective_reward(self.candidate(1.1, 1.2), self.initial),
                           noise.objective_reward(self.candidate(1.2, 1.1), self.initial))
        ratios = noise.objective_ratios(values, self.initial)
        self.assertAlmostEqual(ratios[self.nonstream], 1.1)
        self.assertAlmostEqual(ratios[self.stream], 1.2)

    def test_quality_ttfa_and_continuity_remain_binding(self):
        values = self.candidate(1.1, 1.2)
        self.assertTrue(self.accepted(values))
        for name, key, value in [(self.nonstream,'wer_corpus',.02001),
                                 (self.stream,'wer_corpus',.02001),
                                 (self.stream,'audio_ttfp_p95_s',10.31),
                                 (self.stream,'c50',9.69)]:
            broken = copy.deepcopy(values); broken[name][key] = value
            with self.subTest(workload=name, metric=key): self.assertFalse(self.accepted(broken))

    def test_baseline_chain_uses_bk_for_both_gates_and_b0_for_reward(self):
        with patch.object(noise, 'HERE', ROOT / 'tests'):
            b0 = noise.summarize([calibration()])
            def submission(parent, normal, stream):
                record = calibration()
                record.update(status='eligible', baseline=parent)
                for name, ratio in [(self.nonstream, normal), (self.stream, stream)]:
                    for run in record['runs'][name]:
                        run['metrics']['throughput_qps'] = 10.0 * ratio
                record['means'] = common.means(record['runs'])
                record['reward'] = noise.objective_reward(record['means'], parent['initial_means'])
                return record
            c1 = submission(b0, 1.1, 1.2)
            b1 = noise.freeze_accepted(c1, b0)
            self.assertEqual(b1['initial_means'], b0['means'])
            self.assertEqual(b1['generation'], 1)
            # Above B0 in both modes, but below B1 in streaming: reject.
            with self.assertRaises(common.Failure): noise.freeze_accepted(submission(b1, 1.2, 1.19), b1)
            c2 = submission(b1, 1.2, 1.3)
            self.assertAlmostEqual(c2['reward'], 1.3**(2/3)*1.2**(1/3))
            b2 = noise.freeze_accepted(c2, b1)
            self.assertEqual(b2['generation'], 2)
            self.assertEqual(b2['initial_means'], b0['means'])
            broken = copy.deepcopy(c2); broken['reward'] = 1.2
            with self.assertRaises(common.Failure): noise.freeze_accepted(broken, b1)
            legacy = copy.deepcopy(b1); legacy['means'][self.stream]['throughput_qps'] = 10.0
            legacy['samples'][0]['means'] = legacy['means']
            with self.assertRaises(common.Failure): noise.validate_frozen(legacy, legacy['identity'])

    def test_legacy_policy_and_weights_are_rejected(self):
        for change in [dict(primary_objective=dict(model=self.nonstream, metric='throughput_qps', comparison='>')),
                       dict(reward_weights={self.nonstream: 2/3, self.stream: 1/3}),
                       dict(reward='nonstream-only')]:
            with self.subTest(change=change), patch.dict(common.PROTOCOL['evaluation_policy'], change):
                with self.assertRaises(common.Failure): noise.require_policy_ready()


if __name__ == '__main__': unittest.main()
