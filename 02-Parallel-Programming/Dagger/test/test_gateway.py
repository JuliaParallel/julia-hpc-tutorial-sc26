"""Shared-password transport, authentication boundary, and lease accounting."""
import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'host'))
from gateway_client import Transport, descriptor

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

class GatewayTests(unittest.TestCase):
    def test_host_identity_is_required_and_host_cannot_be_an_ssh_option(self):
        with self.assertRaises(ValueError):
            descriptor('-oProxyCommand=bad')
        with self.assertRaisesRegex(ValueError, 'pinned'):
            descriptor('example.org')
        d = descriptor('example.org', 'ssh-ed25519 AAAA')
        self.assertEqual(d['port'], 22)

    def test_forced_entry_rejects_shell_commands_before_connecting(self):
        for command in ('', 'bash', 'cpu; id', 'odo extra', 'sftp'):
            p = subprocess.run([sys.executable, str(BASE / 'aws/coordinator.py'), 'attach'],
                env=dict(os.environ, SSH_ORIGINAL_COMMAND=command), capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn('shell access is disabled', p.stderr)

    def test_password_uses_private_socket_and_one_transport_for_multiple_commands(self):
        password = 'unit-test-workshop-password'
        real_popen = subprocess.Popen
        count = []
        host = str(BASE / 'host')
        def ssh_or_process(argv, **kwargs):
            if argv[0] == 'ssh':
                count.append(argv)
                self.assertNotIn(password, ' '.join(argv))
                self.assertNotIn(password, '\n'.join(kwargs['env'].values()))
                # Authenticate using the real askpass helper, then simulate the
                # coordinator's worker dispatcher on the same persistent pipe.
                import hashlib
                digest = hashlib.sha256(password.encode()).hexdigest()
                script = ("import hashlib,os,subprocess,sys; "
                    "p=subprocess.check_output([os.environ['SSH_ASKPASS']]); "
                    "assert hashlib.sha256(p).hexdigest()==" + repr(digest) + "; "
                    "sys.path.insert(0," + repr(host) + "); import relay; relay.remote()")
                return real_popen([sys.executable, '-c', script], **kwargs)
            return real_popen(argv, **kwargs)
        transport = Transport(dict(host='example.org', port=22, host_key='ssh-ed25519 AAAA'),
                              'cpu', password, lambda message: None)
        location = transport.root
        try:
            with patch('subprocess.Popen', side_effect=ssh_or_process):
                self.assertEqual(transport.start()['op'], 'ready')
                self.assertEqual(transport._password, '')
                for value in ('one', 'two'):
                    p = subprocess.run(transport.command('printf '+value), capture_output=True, timeout=10)
                    self.assertEqual((p.returncode, p.stdout), (0, value.encode()))
                self.assertEqual(len(count), 1)
                for file in location.iterdir():
                    if file.is_file():
                        self.assertNotIn(password.encode(), file.read_bytes())
                transport.close()
        finally:
            transport.close()
        self.assertFalse(location.exists())

    def test_capacity_counts_pending_reservations_and_preserves_other_instances(self):
        coordinator = module('gateway_coordinator', BASE / 'aws/coordinator.py')
        allocator = object.__new__(coordinator.Allocator)
        allocator.config = dict(threads=8)
        allocator.types = {'c7i.xlarge': 4, 'c7i.2xlarge': 8}
        allocator.leases = {'a': dict(reserved=True, label='visible'),
                            'b': dict(reserved=True, label='not-yet-visible'),
                            'c': dict(reserved=False, label='queued')}
        allocator.ec2 = Mock()
        allocator.ec2.get_paginator.return_value.paginate.return_value = [dict(Reservations=[dict(Instances=[
            dict(InstanceType='c7i.xlarge', Tags=[]),
            dict(InstanceType='c7i.2xlarge', Tags=[dict(Key='vmbench-label', Value='visible')])])])]
        self.assertEqual(allocator.usage(), 20)  # 4 + 8 physical, plus 8 reserved

    def test_queued_disconnect_removes_only_its_lease(self):
        coordinator = module('gateway_cleanup', BASE / 'aws/coordinator.py')
        async def check():
            allocator = object.__new__(coordinator.Allocator)
            allocator.leases = {'a': dict(phase='queued'), 'b': dict(phase='running')}
            allocator.queue = ['a']
            allocator.persist = Mock()
            allocator.rpc = Mock(side_effect=AssertionError('Queued clients have no VM to terminate'))
            await allocator.release('a')
            self.assertEqual(list(allocator.leases), ['b'])
            self.assertEqual(allocator.queue, [])
        asyncio.run(check())

    def test_idle_and_absolute_deadlines_close_sessions_despite_recent_activity(self):
        coordinator = module('gateway_expiry', BASE / 'aws/coordinator.py')
        async def check():
            from unittest.mock import AsyncMock
            for settings, activity, ticks in (
                (dict(max_lifetime=100, idle_timeout=10), [0], [0, 9, 10]),
                (dict(max_lifetime=10, idle_timeout=100), [9], [0, 9, 10]),
            ):
                stopped, status = asyncio.Event(), AsyncMock()
                clock = iter(ticks)
                await coordinator.expire_session(settings, stopped, activity, status,
                    clock=lambda: next(clock), interval=0)
                self.assertTrue(stopped.is_set())
                self.assertEqual(status.await_count, 1)
        asyncio.run(check())

    def test_failed_termination_keeps_capacity_reserved_until_retry_succeeds(self):
        coordinator = module('gateway_retry', BASE / 'aws/coordinator.py')
        async def check():
            from unittest.mock import AsyncMock
            allocator = object.__new__(coordinator.Allocator)
            allocator.leases = {'a': dict(phase='running', label='owned-a', handle='private', reserved=True)}
            allocator.queue = []
            allocator.persist = Mock()
            allocator.rpc = AsyncMock(side_effect=OSError('temporary AWS failure'))
            await allocator.release('a')
            self.assertTrue(allocator.leases['a']['reserved'])
            self.assertEqual(allocator.leases['a']['phase'], 'cleanup_failed')
            allocator.rpc = AsyncMock(return_value={'ok': True})
            await allocator.release('a')
            self.assertEqual(allocator.leases, {})
        asyncio.run(check())


    def test_slow_streams_do_not_block_other_attendees_or_lose_large_reports(self):
        import socket,time,threading,base64
        relay = str(BASE / 'host/relay.py')
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryFile() as log:
            path = directory + '/socket'
            server = subprocess.Popen([sys.executable, relay, 'serve', '--local', '--socket', path], stderr=log)
            stalled = slow = None
            try:
                deadline = time.monotonic() + 10
                while not Path(path).exists():
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(.02)
                stalled = subprocess.Popen([sys.executable, relay, 'client', '--socket', path,
                    '--command', 'sleep 60'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log)
                def fill():
                    try:
                        stalled.stdin.write(b'x' * 1000000)
                        stalled.stdin.flush()
                    except (OSError, ValueError):
                        pass
                threading.Thread(target=fill, daemon=True).start()
                time.sleep(.2)
                def quick():
                    p = subprocess.run([sys.executable, relay, 'client', '--socket', path,
                        '--command', 'printf alive'], capture_output=True, timeout=5)
                    self.assertEqual((p.returncode, p.stdout), (0, b'alive'))
                quick()
                slow = socket.socket(socket.AF_UNIX)
                slow.settimeout(15)
                slow.connect(path)
                slow.sendall((json.dumps(dict(command='head -c 16000000 /dev/zero')) + '\n').encode())
                time.sleep(.3)  # Let the output queue apply backpressure.
                quick()
                count = 0
                with slow.makefile('rb') as stream:
                    while True:
                        msg = json.loads(stream.readline())
                        if msg['op'] == 'exit':
                            self.assertEqual(msg['returncode'], 0)
                            break
                        count += len(base64.b64decode(msg['data']))
                self.assertEqual(count, 16000000)
            finally:
                if stalled:
                    stalled.terminate()
                    stalled.communicate(timeout=5)
                if slow:
                    slow.close()
                server.terminate()
                server.wait(timeout=35)

if __name__ == '__main__':
    unittest.main()
