"""CPU regressions for task-owned scheduling, never model/GPU execution."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
TEST_GPUS = [f"GPU-00000000-0000-0000-0000-{i:012d}" for i in range(4)]
BINDING = {"adapter_sha256": "fixture-digest", "baseline": False, "adapter_path": ".", "config_sha256": "config", "manifest_sha256": "manifest"}


def cpu_worker_target(connection, index, gpu):
    """Real spawned worker/IPC; replace only unavailable model and asset boundary."""
    import os
    from types import SimpleNamespace
    import parallel_encoder as m
    # Exercise precisely the main-file re-execution used by multiprocessing
    # spawn, not merely importing the Torch-free helper under unittest's main.
    import runpy
    runpy.run_path(str(Path(__file__).with_name("evaluate.py")), run_name="__mp_main__")
    if "torch" in sys.modules:
        raise AssertionError("formal spawn entry imported Torch before worker visibility")
    def evaluator():
        if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu or "torch" in sys.modules:
            raise AssertionError("GPU isolation was not established before evaluator load")
        cuda = SimpleNamespace(device_count=lambda: 1, set_device=lambda index: None,
            reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 64 * 1024**2)
        def encode(model, texts, instruction, maximum):
            result = np.zeros((len(texts), 4096), dtype=np.float32)
            result[:, 0] = [float(text) for text in texts]
            return result
        return SimpleNamespace(torch=SimpleNamespace(cuda=cuda), load_base_model=lambda: object(),
                               load_candidate=lambda base, adapter: base, encode=encode)
    m.load_evaluator = evaluator
    m.validate_candidate = lambda e, canonical=False: (Path("."), BINDING)
    m.recheck_binding = lambda e, adapter, binding: None
    m.worker_main(connection, index, gpu, BINDING, False)


def load_helper(test):
    path = Path(__file__).with_name("parallel_encoder.py")
    test.assertTrue(path.is_file(), "parallel encoder implementation is absent")
    spec = importlib.util.spec_from_file_location("parallel_encoder", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Pipe:
    def __init__(self, worker):
        self.worker = worker
        self.pending = []
        self.jobs = []
    def send(self, job):
        # This double replaces only GPU inference; scheduler/validation are real.
        if self.pending:
            raise AssertionError("more than one in-flight job per worker")
        self.jobs.append(job)
        values = np.array([[float(text), -float(text)] for text in job["texts"]], dtype=np.float32)
        self.pending.append({"op": "encoded", "call_id": job["call_id"], "offset": job["offset"],
                             "count": len(job["texts"]), "embedding": values, "worker": self.worker})
    def recv(self):
        return self.pending.pop(0)


class ParallelEncoderTests(unittest.TestCase):
    def test_expired_pool_rejects_even_an_immediately_ready_reply(self):
        # Break: ready-pipe fast path bypasses the bounded pool deadline.
        m = load_helper(self)
        pool = m.ParallelEncoder(TEST_GPUS, BINDING)
        pool.deadline = 0
        from types import SimpleNamespace
        reply = SimpleNamespace(poll=lambda timeout: True, recv=lambda: {"op": "ready"})
        with self.assertRaises(TimeoutError): pool.receive(reply)

    def test_reassembles_out_of_order_results_and_bounds_inflight(self):
        # Break caught: concatenating arrival order, dropping tail, unbounded submission.
        m = load_helper(self)
        pipes = [Pipe(0), Pipe(1)]
        result, counts = m.dispatch_chunks(["1", "2", "3", "4", "5"], "", 2048,
            pipes, 7, lambda choices, timeout: list(reversed(choices)), lambda: None,
            chunk_size=2, dimension=2)
        self.assertEqual(result.tolist(), [[1, -1], [2, -2], [3, -3], [4, -4], [5, -5]])
        self.assertEqual(sum(counts), 5)
        self.assertTrue(all(job["call_id"] == 7 and job["metric"] for p in pipes for job in p.jobs))

    def test_rejects_duplicate_gap_stale_shape_dtype_and_nonfinite(self):
        # Break caught: accepting malformed or incomplete worker buffers as a metric.
        m = load_helper(self)
        good = np.array([[1, 2], [3, 4]], dtype=np.float32)
        a = m.ChunkAssembler(4, 9, dimension=2)
        a.accept(9, 0, 2, good)
        with self.assertRaises(RuntimeError):
            a.accept(9, 0, 2, good)
        with self.assertRaises(RuntimeError):
            a.finish()
        for call, offset, count, data in [(8, 2, 2, good), (9, 3, 2, good),
                (9, 2, 1, good), (9, 2, 2, good.astype(np.float64)),
                (9, 2, 2, np.full((2, 2), np.nan, dtype=np.float32))]:
            with self.subTest(call=call, offset=offset, count=count), self.assertRaises(RuntimeError):
                a.accept(call, offset, count, data)
        a.accept(9, 2, 2, good)
        self.assertEqual(a.finish().tolist(), [[1, 2], [3, 4], [1, 2], [3, 4]])

    def test_scheduler_rejects_wrong_worker_offset_and_worker_error(self):
        # Break caught: trusted transport delivering stale/wrong assignment or failure.
        m = load_helper(self)
        for alteration in ({"worker": 1}, {"worker": False}, {"offset": 1}, {"call_id": 8}, {"op": "error", "error": "OOM"}):
            pipe = Pipe(0)
            original = pipe.recv
            def recv():
                result = original()
                result.update(alteration)
                return result
            pipe.recv = recv
            with self.subTest(alteration=alteration), self.assertRaises(RuntimeError):
                m.dispatch_chunks(["1"], "", 2048, [pipe], 7,
                    lambda choices, timeout: choices, lambda: None, dimension=2)

    def test_gpu_visibility_is_fixed_before_evaluator_loading(self):
        # Break caught: importing Torch with all cards visible, silently increasing batch.
        m = load_helper(self)
        from unittest.mock import patch
        import os
        with patch.dict(os.environ, {}, clear=True):
            m.configure_worker_environment(TEST_GPUS[0])
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], TEST_GPUS[0])
        for invalid in ([], TEST_GPUS[:-1], [TEST_GPUS[0]] * 4, ["0", "1", "2", "3"]):
            with self.assertRaises(RuntimeError):
                m.validate_gpu_uuids(invalid)

    def test_selects_only_four_authorized_devices_resolves_local_ordinals(self):
        # Break: hardcoded cards, host index assumptions, duplicates, visibility escape.
        m = load_helper(self)
        self.assertTrue(hasattr(m, "select_gpu_uuids"), "portable device selection missing")
        inventory = [(str(i + 4), gpu, f"0000:{i + 10:02x}:00.0") for i, gpu in enumerate(TEST_GPUS)]
        self.assertEqual(m.select_gpu_uuids(inventory, {}), TEST_GPUS)
        self.assertEqual(m.select_gpu_uuids(inventory, {"NVIDIA_VISIBLE_DEVICES": "4,5,6,7", "CUDA_VISIBLE_DEVICES": "3,2,1,0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}), TEST_GPUS[::-1])
        self.assertEqual(m.select_gpu_uuids(inventory, {"CUDA_VISIBLE_DEVICES": ",".join(TEST_GPUS)}), TEST_GPUS)
        for env in ({"CUDA_VISIBLE_DEVICES": "0,1,2"}, {"CUDA_VISIBLE_DEVICES": "0,0,1,2"},
                    {"CUDA_VISIBLE_DEVICES": ""}, {"NVIDIA_VISIBLE_DEVICES": "none"},
                    {"CUDA_VISIBLE_DEVICES": "4,5,6,7"}):
            with self.subTest(env=env), self.assertRaises(RuntimeError): m.select_gpu_uuids(inventory, env)

    def test_numeric_mask_rejects_ambiguous_order_with_more_than_four_devices(self):
        # Break: PCI sorting silently changes the incoming FASTEST_FIRST GPU set.
        m = load_helper(self)
        fifth = "GPU-00000000-0000-0000-0000-000000000004"
        inventory = [(str(i), gpu, f"0000:{4-i:02x}:00.0") for i, gpu in enumerate(TEST_GPUS + [fifth])]
        for order in (None, "FASTEST_FIRST", "invalid"):
            env = {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}
            if order is not None: env["CUDA_DEVICE_ORDER"] = order
            with self.subTest(order=order), self.assertRaisesRegex(RuntimeError, "PCI_BUS_ID.*UUID"):
                m.select_gpu_uuids(inventory, env)
        self.assertEqual(m.select_gpu_uuids(inventory, {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}),
                         [fifth, TEST_GPUS[3], TEST_GPUS[2], TEST_GPUS[1]])
        self.assertEqual(m.select_gpu_uuids(inventory, {"CUDA_VISIBLE_DEVICES": ",".join(TEST_GPUS), "CUDA_DEVICE_ORDER": "FASTEST_FIRST"}), TEST_GPUS)

    def test_cdi_void_uses_exposed_inventory_without_broadening_cuda_mask(self):
        # Break: CDI's post-injection void marker is mistaken for a GPU selector.
        m = load_helper(self)
        inventory = [(str(i), gpu, f"0000:{i:02x}:00.0") for i, gpu in enumerate(TEST_GPUS)]
        for marker in ("void", "", None):
            env = {"CUDA_VISIBLE_DEVICES": ",".join(TEST_GPUS), "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
            if marker is not None: env["NVIDIA_VISIBLE_DEVICES"] = marker
            with self.subTest(marker=marker):
                self.assertEqual(m.select_gpu_uuids(inventory, env), TEST_GPUS)
        outside = "GPU-00000000-0000-0000-0000-000000000099"
        for mask in ("", ",".join(TEST_GPUS[:3]), ",".join(TEST_GPUS[:3] + [outside])):
            with self.subTest(mask=mask), self.assertRaises(RuntimeError):
                m.select_gpu_uuids(inventory, {"NVIDIA_VISIBLE_DEVICES": "void", "CUDA_VISIBLE_DEVICES": mask})
        with self.assertRaises(RuntimeError):
            m.select_gpu_uuids(inventory, {"NVIDIA_VISIBLE_DEVICES": "none", "CUDA_VISIBLE_DEVICES": ",".join(TEST_GPUS)})
        with self.assertRaises(RuntimeError):
            m.select_gpu_uuids([], {"NVIDIA_VISIBLE_DEVICES": "void", "CUDA_VISIBLE_DEVICES": ",".join(TEST_GPUS)})

    def test_scheduler_rechecks_deadline_after_wait_and_receive(self):
        # Break: an expired final response starts retrieval before the next check.
        m = load_helper(self)
        for expire_at in ("wait", "recv"):
            now = [0]
            pipe = Pipe(0)
            original = pipe.recv
            def recv():
                if expire_at == "recv": now[0] = 4500
                return original()
            pipe.recv = recv
            def ready(choices, timeout):
                if expire_at == "wait": now[0] = 4500
                return choices
            def check():
                if now[0] >= 4500: raise TimeoutError("deadline")
            with self.subTest(expire_at=expire_at), self.assertRaises(TimeoutError):
                m.dispatch_chunks(["1"], "", 2048, [pipe], 7, ready, check, dimension=2)

    def test_bitwise_consistency_rejects_signed_zero_and_nan(self):
        # Break caught: tolerance-based equality or NaN accepting a different protocol.
        m = load_helper(self)
        good = np.array([[0.0, 1.0]], dtype=np.float32)
        m.require_bitwise_equal(good, good.copy())
        for other in (np.array([[-0.0, 1.0]], dtype=np.float32),
                      np.array([[0.0, np.nan]], dtype=np.float32), good.astype(np.float64)):
            with self.assertRaises(RuntimeError):
                m.require_bitwise_equal(good, other)

    def test_cleanup_attempts_every_worker_even_when_one_cannot_be_stopped(self):
        # Break caught: one cleanup failure leaves the remaining GPU workers alive.
        m = load_helper(self)
        class Process:
            def __init__(self, stuck):
                self.alive = True
                self.stuck = stuck
                self.terminated = self.killed = False
            def is_alive(self): return self.alive
            def terminate(self): self.terminated = True
            def join(self, timeout): pass
            def kill(self):
                self.killed = True
                self.alive = self.stuck
        pool = m.ParallelEncoder(TEST_GPUS, BINDING)
        pool.processes = [Process(True), Process(False)]
        with self.assertRaises(RuntimeError):
            pool.close()
        self.assertTrue(pool.processes[1].terminated)
        self.assertTrue(pool.processes[1].killed)
        self.assertFalse(pool.processes[1].alive)

    def test_real_pool_synthetic_check_uses_nonempty_inputs_every_worker_both_protocols(self):
        # Break caught: empty prompt-masked pooling, partial consistency checks,
        # or counting synthetic work. Same-image CPU tokenizer proof is retained.
        m = load_helper(self)
        class SyntheticPipe:
            def __init__(self, index, fail):
                self.index, self.fail, self.jobs = index, fail, []
            def send(self, job):
                if not all(text.strip() for text in job["texts"]):
                    raise AssertionError("synthetic fixtures must retain content under prompt masking")
                self.jobs.append(job)
            def poll(self, timeout): return True
            def recv(self):
                job = self.jobs[-1]
                array = np.zeros((3, 4096), dtype=np.float32)
                if self.fail: array[0, 0] = 1
                return {"op": "encoded", "worker": self.index, "call_id": job["call_id"],
                        "offset": 0, "count": 3, "embedding": array}
        pool = m.ParallelEncoder(TEST_GPUS, BINDING)
        pool.connections = [SyntheticPipe(i, False) for i in range(4)]
        result = pool.consistency_check()
        self.assertEqual(result, {"passed": True, "bitwise_equal": True, "worker_count": 4,
                                  "text_count": 3, "protocol_count": 2, "reference_worker": 0})
        self.assertEqual([len(p.jobs) for p in pool.connections], [4, 2, 2, 2])
        self.assertTrue(all(not j["metric"] for p in pool.connections for j in p.jobs))
        self.assertEqual(pool.total_metric_encodes, 0)
        pool.connections[3].fail = True
        with self.assertRaises(RuntimeError):
            pool.consistency_check()

    def test_spawned_cpu_worker_isolates_visibility_counts_metrics_and_redacts_failure(self):
        # Break caught: spawn/import visibility regression, wrong worker accounting,
        # or including input-bearing exception text in coordinator logs.
        m = load_helper(self)
        import multiprocessing
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=cpu_worker_target, args=(child, 0, TEST_GPUS[0]))
        process.start()
        child.close()
        try:
            self.assertTrue(parent.poll(10))
            ready = parent.recv()
            self.assertEqual((ready["op"], ready["gpu_uuid"], ready["pid"]),
                             ("ready", TEST_GPUS[0], process.pid))
            for metric in (False, True):
                parent.send({"op": "encode", "call_id": 1, "offset": 0, "texts": ["1", "2", "3"],
                             "instruction": "", "max_length": 2048, "metric": metric})
                self.assertTrue(parent.poll(10))
                reply = parent.recv()
                self.assertEqual(reply["embedding"][:, 0].tolist(), [1, 2, 3])
            parent.send({"op": "stats"})
            self.assertTrue(parent.poll(10))
            self.assertEqual(parent.recv()["stats"]["metric_encodes"], 3)
            parent.send({"op": "encode", "call_id": 2, "offset": 0, "texts": ["SECRET_CASE_CONTENT"],
                         "instruction": "", "max_length": 2048, "metric": True})
            self.assertTrue(parent.poll(10))
            failure = parent.recv()
            self.assertEqual(failure, {"op": "error", "worker": 0, "stage": "encoding", "error": "ValueError"})
            process.join(timeout=10)
            self.assertEqual(process.exitcode, 1)
        finally:
            parent.close()
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
