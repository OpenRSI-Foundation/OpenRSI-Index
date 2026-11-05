"""Non-streaming scope, throughput acceptance and baseline-chain regressions."""
import copy
import unittest
from unittest.mock import patch

from test_model_lane import ROOT, calibration, common, noise, runs


class NonstreamTests(unittest.TestCase):
    def test_only_nonstreaming_requests_are_scheduled(self):
        cases = common.evaluation_cases()
        self.assertEqual(len(cases), 1)
        case = cases[0]
        self.assertFalse(case['stream'])
        self.assertNotIn('--stream', case['argv'])
        self.assertEqual(case['argv'], common.PROTOCOL['lanes'][0]['argv'])
        invalid = runs()
        invalid['ming_stream'] = copy.deepcopy(invalid['ming'])
        with self.assertRaises(common.Failure):
            common.validate_runs(invalid, 'synthetic')

    def test_unchanged_or_slower_throughput_is_rejected(self):
        baseline = {'means': common.means(runs())}
        for throughput, accepted in [(9.9, False), (10, False), (10.1, True)]:
            candidate = copy.deepcopy(baseline['means'])
            candidate['ming']['throughput_qps'] = throughput
            guards = noise.compare(candidate, baseline)[1]
            self.assertEqual(all(all(g.values()) for g in guards.values()), accepted)
            self.assertAlmostEqual(noise.objective_reward(candidate, baseline['means']), throughput / 10)

    def test_rolling_acceptance_uses_bk_and_reward_retains_b0(self):
        with patch.object(noise, 'HERE', ROOT / 'tests'):
            initial = noise.summarize([calibration()])
            def submission(parent, throughput):
                record = calibration()
                record.update(status='eligible', baseline=parent)
                for run in record['runs']['ming']:
                    run['metrics']['throughput_qps'] = throughput
                record['means'] = common.means(record['runs'])
                record['reward'] = noise.objective_reward(record['means'], parent['initial_means'])
                return record
            current = noise.freeze_accepted(submission(initial, 11), initial)
            with self.assertRaises(common.Failure):
                noise.freeze_accepted(submission(current, 10.9), current)
            second = submission(current, 12)
            self.assertAlmostEqual(second['reward'], 1.2)
            advanced = noise.freeze_accepted(second, current)
            self.assertEqual(advanced['initial_means'], initial['means'])
            self.assertEqual(advanced['generation'], 2)

    def test_old_dual_mode_policy_is_rejected(self):
        with patch.dict(common.PROTOCOL['evaluation_policy'], reward_weights={'ming': 1/3, 'ming_stream': 2/3}):
            with self.assertRaises(common.Failure):
                noise.require_policy_ready()
