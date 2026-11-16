"""Local execution uses isolated sessions, CPU controls and shell-free launches."""
import argparse
import io
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'host'))
import local
import remote_agent
spec = importlib.util.spec_from_file_location('local_console_app', BASE / 'host/app.py')
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


class LocalTests(unittest.TestCase):
    def test_connect_never_calls_ssh_and_isolates_sessions(self):
        with tempfile.TemporaryDirectory(prefix='local sessions ') as folder:
            args = argparse.Namespace(local=True, backend='cpu', threads=2, remote_root=folder)
            c = app.Console(args)
            c._read = Mock()
            with patch.object(app.subprocess, 'run', side_effect=AssertionError('SSH called')), patch.object(app.subprocess, 'Popen') as popen:
                c._connect()
                command = popen.call_args.args[0]
                options = popen.call_args.kwargs
                self.assertEqual(command[0], sys.executable)
                self.assertEqual(command[-1], 'serve')
                self.assertEqual(options['env']['SC26_LOCAL'], '1')
                self.assertEqual(options['env']['JULIA_NUM_THREADS'], '2')
                session = Path(options['cwd'])
                self.assertTrue((session / 'host/local.py').is_file())
                self.assertTrue((session / 'cpu/Project.toml').is_file())
                self.assertFalse((session / 'cpu/Manifest.toml').exists())
                c._connect()
                self.assertNotEqual(session, popen.call_args.kwargs['cwd'])
            config = c.configuration()
            self.assertTrue(config['local'])
            self.assertFalse(config['gateway'])
            self.assertEqual(config['targets'], {'cpu': 'CPU'})
            self.assertEqual(config['max_ranks'], 2)
            self.assertNotIn('03_gpu', [d[0] for d in config['demos']])
            self.assertEqual(c.ready_message(), 'Ready · local CPU')

    def test_launcher_preserves_paths_and_uses_julia_mpi_binary(self):
        with patch.dict(os.environ, JULIA_PROJECT='/a project/cpu', SC26_CPU_MPI='0', SC26_THREADS='1', JULIA_NUM_THREADS='8'), patch.object(local, 'julia', return_value='/some Julia/julia'), patch.object(local.os, 'execv') as execute:
            local.launch(1, ['--startup-file=no', '/a worker/worker.jl'])
            self.assertEqual(os.environ['JULIA_NUM_THREADS'], '1')
            self.assertEqual(execute.call_args.args[1], ['/some Julia/julia', '--startup-file=no', '--project=/a project/cpu', '--startup-file=no', '/a worker/worker.jl'])
            os.environ['SC26_CPU_MPI'] = '1'
            local.launch(2, ['/a worker/worker.jl', '--mpi'])
            argv = execute.call_args.args[1]
            self.assertIn('MPI.mpiexec()', argv[4])
            self.assertEqual(argv[-3:], ['2', '/a worker/worker.jl', '--mpi'])

    def test_local_cli_rejects_remote_options_before_authentication(self):
        for args in (['--gateway'], ['--ssh-host', 'example'], ['--via-relay']):
            result = subprocess.run([sys.executable, str(BASE / 'host/app.py'), '--local', *args], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('--local cannot be combined', result.stderr)

    def test_missing_julia_has_actionable_error(self):
        with patch.object(local.shutil, 'which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'Julia 1.11'):
                local.julia()

    def test_disconnect_stops_preparation_before_it_can_read_quit(self):
        c = app.Console(argparse.Namespace(local=True))
        proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                                stdin=subprocess.PIPE, text=True, start_new_session=True)
        c.process, c.connecting = proc, True
        try:
            c.disconnect()
            self.assertIsNotNone(proc.poll())
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_intentional_disconnect_does_not_report_a_crash(self):
        c = app.Console(argparse.Namespace(local=True))
        c.stop_event.set()
        proc = Mock(stdout=[])
        proc.wait.return_value = -15
        c._read(proc)
        self.assertEqual(c.events, [])

    def test_next_run_is_allowed_when_completion_is_reported(self):
        events = []
        proc = Mock(stdout=io.StringIO(remote_agent.PREFIX + json.dumps(dict(kind='done', ok=True)) + '\n'))
        def wait():
            self.assertFalse(any(e['kind'] == 'done' for e in events))
            return 0
        proc.wait.side_effect = wait
        run = dict(proc=proc, id='local-test', done=False, cancelled=False)
        def emit(kind, **fields):
            if kind == 'done':
                self.assertIsNone(remote_agent.current)
            events.append(dict(kind=kind, **fields))
        with patch.object(remote_agent, 'LOCAL', True), patch.object(remote_agent, 'current', run), patch.object(remote_agent, 'emit', side_effect=emit):
            remote_agent.relay(run)
        self.assertEqual([e['kind'] for e in events], ['worker_exit', 'done'])


if __name__ == '__main__':
    unittest.main()
