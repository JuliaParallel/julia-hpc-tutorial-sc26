"""Fallback boundaries and concurrent streams; no AWS or Odo required."""
import argparse
import concurrent.futures
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest
from unittest.mock import Mock

BASE = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

app = load('fallback_console', BASE / 'host/app.py')

class FallbackTests(unittest.TestCase):
    def test_cpu_api_rejects_unavailable_hardware_before_sending(self):
        c = app.Console(argparse.Namespace(backend='cpu', threads=8))
        c.send = Mock()
        with self.assertRaisesRegex(ValueError, 'no GPUs'):
            c.run(dict(op='demo', demo='03_gpu'))
        with self.assertRaisesRegex(ValueError, 'at most 8'):
            c.run(dict(op='demo', demo='06_multinode', ranks=9))
        c.send.assert_not_called()
        self.assertEqual(c.configuration()['targets'], {'cpu': 'CPU'})
        self.assertNotIn('03_gpu', [d[0] for d in c.configuration()['demos']])
        c.run(dict(op='demo', demo='06_multinode', ranks=2))
        sent = c.send.call_args.args[0]
        self.assertEqual((sent['mode'], sent['ranks'], sent['gpus']), ('mpi', 2, 0))

    def test_cpu_coordinator_splits_cores_and_cancels_without_slurm(self):
        agent = load('cpu_agent', BASE / 'host/remote_agent.py')
        agent.CPU, agent.CPU_THREADS = True, 8
        agent.emit = Mock()
        with self.assertRaisesRegex(ValueError, 'no GPUs'):
            agent.resources(dict(gpus=1))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'cpu').mkdir()
            (root / 'cpu/launch.sh').write_text(
                'echo "$SC26_THREADS $SC26_CPU_MPI"; cat >/dev/null; sleep 60 & wait $!\n')
            agent.ROOT = root
            agent.execute(dict(id='a', mode='mpi', ranks=2, op='eval'))
            run = agent.current
            self.assertIsNotNone(run)
            deadline = time.monotonic() + 5
            while not any(c.args == ('output',) for c in agent.emit.call_args_list):
                self.assertLess(time.monotonic(), deadline)
                time.sleep(.01)
            agent.cancel()
            self.assertIsNotNone(run['proc'].poll())
            self.assertTrue(run['cancelled'])
        self.assertTrue(any(c.args == ('output',) and c.kwargs['text'] == '4 1\n'
                            for c in agent.emit.call_args_list))


class RelayTests(unittest.TestCase):
    def test_concurrent_binary_streams_exit_codes_and_disconnect_isolation(self):
        relay = str(BASE / 'host/relay.py')
        with tempfile.TemporaryDirectory() as directory:
            sock = directory + '/socket'
            with tempfile.TemporaryFile() as log:
                server = subprocess.Popen(['python3', relay, 'serve', '--local', '--socket', sock],
                                          stderr=log)
                sleeper = None
                try:
                    deadline = time.monotonic() + 10
                    while not Path(sock).exists():
                        self.assertIsNone(server.poll())
                        self.assertLess(time.monotonic(), deadline)
                        time.sleep(.02)
                    def command(script, data=b''):
                        return subprocess.run(['python3', relay, 'client', '--socket', sock,
                            '--command', script], input=data, capture_output=True, timeout=20)
                    # Each stream has a different payload, larger than one frame.
                    def roundtrip(i):
                        data = bytes([i]) + os.urandom(160000)
                        p = command('cat', data)
                        self.assertEqual(p.returncode, 0, p.stderr)
                        self.assertEqual(p.stdout, data)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
                        list(pool.map(roundtrip, range(12)))
                    p = command('printf failed; exit 7')
                    self.assertEqual((p.returncode, p.stdout), (7, b'failed'))
                    marker = Path(directory) / 'child-pid'
                    sleeper = subprocess.Popen(['python3', relay, 'client', '--socket', sock,
                        '--command', 'echo $$ > "{}"; exec sleep 60'.format(marker)],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log)
                    deadline = time.monotonic() + 10
                    while not marker.exists():
                        self.assertLess(time.monotonic(), deadline)
                        time.sleep(.02)
                    pid = int(marker.read_text())
                    sleeper.terminate()
                    sleeper.wait(timeout=5)
                    deadline = time.monotonic() + 10
                    while True:
                        try:
                            os.kill(pid, 0)
                        except ProcessLookupError:
                            break
                        self.assertLess(time.monotonic(), deadline, 'Disconnected client left its process running')
                        time.sleep(.02)
                    self.assertEqual(command('printf survivor').stdout, b'survivor')
                finally:
                    if sleeper:
                        if sleeper.poll() is None:
                            sleeper.kill()
                            sleeper.wait()
                        sleeper.stdin.close()
                        sleeper.stdout.close()
                    server.terminate()
                    server.wait(timeout=35)
                self.assertFalse(Path(sock).exists())

if __name__ == '__main__':
    unittest.main()
