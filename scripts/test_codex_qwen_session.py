import importlib.util
from pathlib import Path
import subprocess
import os
import socket
import time
import tempfile
import unittest
from unittest.mock import patch, Mock

spec = importlib.util.spec_from_file_location('session', Path(__file__).with_name('codex_qwen_session.py'))
session = importlib.util.module_from_spec(spec)
spec.loader.exec_module(session)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.args = session.parser().parse_args(['supervise', '--state-dir', self.directory.name])
        self.app = (42, session.APP + ' --user-data-dir=' + self.args.profile)
        self.server = Mock(pid=900)
        self.server.poll.return_value = None
        self.spawn = Mock(return_value=self.server)
        self.run = Mock(return_value=Mock(returncode=0))

    def execute(self, rows, probes, **kwargs):
        with patch.object(session, 'stop_group') as stop:
            result = session.supervise(self.args, rows=Mock(side_effect=rows),
                probe=Mock(side_effect=probes), popen=self.spawn, run=self.run,
                sleep=lambda _: None, **kwargs)
            stop.assert_called_once_with(self.server)
            return result

    def test_launch_wait_and_shutdown(self):
        self.assertEqual(self.execute([[], [self.app], [self.app], []], [False, True]), 0)
        command = self.run.call_args.args[0]
        self.assertIn('--user-data-dir=' + self.args.profile, command)
        self.assertTrue(self.spawn.call_args.kwargs['start_new_session'])

    def test_other_profiles_do_not_keep_server_running(self):
        other = (43, session.APP + ' --user-data-dir=/other/profile')
        self.execute([[], [self.app], [other]], [False, True])

    def test_open_failure_stops_server(self):
        self.run.return_value.returncode = 1
        with patch.object(session, 'stop_group') as stop:
            with self.assertRaisesRegex(RuntimeError, 'launch failed'):
                session.supervise(self.args, rows=lambda: [], probe=Mock(side_effect=[False, True]),
                                  popen=self.spawn, run=self.run)
            stop.assert_called_once_with(self.server)

    def test_readiness_failure_stops_server(self):
        self.server.poll.return_value = 1
        with patch.object(session, 'stop_group') as stop:
            with self.assertRaisesRegex(RuntimeError, 'before becoming ready'):
                session.supervise(self.args, rows=lambda: [], probe=lambda _: False, popen=self.spawn)
            stop.assert_called_once_with(self.server)

    def test_adoption_does_not_reopen_app_or_spawn_server(self):
        self.args.adopt_existing = True
        self.args.adopt_server_pid = 71
        server = (71, self.args.ollama + ' serve')
        rows = Mock(side_effect=[[self.app], [self.app, server], [self.app], [], []])
        session.supervise(self.args, rows=rows, probe=lambda _: True, popen=self.spawn,
                          run=self.run, sleep=lambda _: None)
        self.spawn.assert_not_called()
        self.run.assert_not_called()

    def test_singleton_does_not_spawn_second_server(self):
        lock_path = Path(self.directory.name) / 'session.lock'
        with lock_path.open('a+') as lock:
            session.fcntl.flock(lock, session.fcntl.LOCK_EX)
            self.assertEqual(session.supervise(self.args, popen=self.spawn, rows=lambda: [self.app], probe=lambda _: True), 0)
        self.spawn.assert_not_called()

    def test_adopted_app_early_exit_retires_only_matching_jobs(self):
        self.args.adopt_existing = True
        self.args.adopt_server_pid = 71
        self.args.retire_job = ['owned:71', 'changed:72']
        server = (71, self.args.ollama + ' serve')
        rows = Mock(side_effect=[[self.app], [self.app, server], [], []])
        run = Mock(side_effect=[Mock(returncode=0, stdout='  pid = 71\n'),
                               Mock(returncode=0),
                               Mock(returncode=0, stdout='  pid = 999\n')])
        session.supervise(self.args, rows=rows, probe=lambda _: True,
                          popen=self.spawn, run=run)
        bootouts = [call.args[0] for call in run.call_args_list if 'bootout' in call.args[0]]
        self.assertEqual(bootouts, [['/bin/launchctl', 'bootout',
                                  'gui/%s/owned' % os.getuid()]])
        self.spawn.assert_not_called()

    def test_detached_start_reports_failure(self):
        result = subprocess.run([session.sys.executable, session.__file__, 'start',
            '--state-dir', self.directory.name, '--ollama', '/missing/ollama',
            '--endpoint', 'http://127.0.0.1:1'], capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 1)
        self.assertIn('startup failed', result.stderr)

    def test_relaunch_acquires_lock_after_previous_supervisor_exits(self):
        lock_path = Path(self.directory.name) / 'session.lock'
        with lock_path.open('a+') as previous:
            session.fcntl.flock(previous, session.fcntl.LOCK_EX)
            sleeps = []
            def release_previous(_):
                sleeps.append(True)
                session.fcntl.flock(previous, session.fcntl.LOCK_UN)
            with patch.object(session, 'stop_group') as stop:
                result = session.supervise(self.args,
                    rows=Mock(side_effect=[[], [self.app], []]),
                    probe=Mock(side_effect=[False, False, True]),
                    popen=self.spawn, run=self.run, sleep=release_previous)
                stop.assert_called_once_with(self.server)
        self.assertEqual(result, 0)
        self.assertEqual(len(sleeps), 1)
        self.spawn.assert_called_once()
        self.run.assert_called_once()

    def test_real_server_and_runner_exit_with_profile(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
        self.args.endpoint = 'http://127.0.0.1:%s' % port
        self.args.poll_interval = 0.02
        runner_file = Path(self.directory.name) / 'runner.pid'
        executable = Path(self.directory.name) / 'ollama'
        executable.write_text("#!/usr/bin/env python3\n"
            "import http.server, os, subprocess, sys\n"
            "runner = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
            "open(%r, 'w').write(str(runner.pid))\n" % str(runner_file) +
            "http.server.HTTPServer(('127.0.0.1', %s), http.server.SimpleHTTPRequestHandler).serve_forever()\n" % port)
        executable.chmod(0o755)
        self.args.ollama = str(executable)
        spawned = []
        def spawn(*args, **kwargs):
            process = subprocess.Popen(*args, **kwargs)
            spawned.append(process)
            return process
        def probe(endpoint):
            try:
                with session.urllib.request.urlopen(endpoint, timeout=0.2):
                    return True
            except OSError:
                return False
        session.supervise(self.args, rows=Mock(side_effect=[[], [self.app], [self.app], []]),
                          probe=probe, popen=spawn, run=self.run)
        self.assertIsNotNone(spawned[0].poll())
        runner_pid = int(runner_file.read_text())
        # Killed child may briefly remain a zombie until the OS reaps it.
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            result = subprocess.run(['ps', '-p', str(runner_pid), '-o', 'stat='],
                                    text=True, capture_output=True)
            if not result.stdout.strip() or result.stdout.strip().startswith('Z'):
                break
            time.sleep(0.02)
        else:
            self.fail('Owned runner remained alive after app exit')

    def test_stop_group_kills_runner_after_server_exits(self):
        with patch.object(session.os, 'killpg') as kill:
            session.stop_group(self.server)
        self.assertEqual([call.args[1] for call in kill.call_args_list],
                         [session.signal.SIGTERM, session.signal.SIGKILL])


if __name__ == '__main__':
    unittest.main()
