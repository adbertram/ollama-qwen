#!/usr/bin/env python3
"""Run Ollama only for the lifetime of the dedicated Codex Qwen app."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import tempfile
import urllib.request

APP = '/Applications/ChatGPT.app/Contents/MacOS/ChatGPT'
PROFILE = str(Path.home() / 'Library/Application Support/Codex-Profiles/qwen')


def process_rows():
    output = subprocess.check_output(['ps', '-axo', 'pid=,command='], text=True)
    return [(int(line.strip().split(None, 1)[0]), line.strip().split(None, 1)[1])
            for line in output.splitlines() if len(line.strip().split(None, 1)) == 2]


def qwen_pids(rows, profile):
    # ps does not quote spaces. Match the complete known executable and argument.
    expected = APP + ' --user-data-dir=' + profile
    return [pid for pid, command in rows if command == expected]


def healthy(endpoint):
    try:
        with urllib.request.urlopen(endpoint + '/api/version', timeout=1) as response:
            return response.status == 200
    except (OSError, ValueError):
        return False


def stop_group(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        pass
    # Runners can outlive the server; always finish the owned process group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def supervise(args, *, rows=process_rows, probe=healthy, popen=subprocess.Popen,
              run=subprocess.run, sleep=time.sleep, now=time.monotonic):
    state = Path(args.state_dir)
    state.mkdir(parents=True, exist_ok=True)
    with (state / 'session.lock').open('a+') as lock:
        deadline = now() + args.start_timeout
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if probe(args.endpoint) and qwen_pids(rows(), args.profile):
                    if args.ready_file:
                        Path(args.ready_file).write_text('ready')
                    return 0
                if now() >= deadline:
                    raise RuntimeError('Existing supervisor did not become ready')
                sleep(args.poll_interval)
        server = None
        app_closed = False
        adopted_identity = None
        try:
            existing = qwen_pids(rows(), args.profile)
            if args.adopt_existing and not existing:
                raise RuntimeError('No running Codex Qwen profile to adopt')
            if args.adopt_server_pid:
                if not args.adopt_existing:
                    raise RuntimeError('--adopt-server-pid requires --adopt-existing')
                adopted_identity = next((command for pid, command in rows() if pid == args.adopt_server_pid), None)
                if adopted_identity != args.ollama + ' serve' or not probe(args.endpoint):
                    raise RuntimeError('Adopted Ollama PID or endpoint does not match the expected server')
            elif probe(args.endpoint):
                raise RuntimeError('Ollama endpoint already occupied; stop its existing owner before starting supervisor')
            env = os.environ.copy()
            env.update(OLLAMA_HOST=args.endpoint.removeprefix('http://'),
                       OLLAMA_FLASH_ATTENTION='1', OLLAMA_KV_CACHE_TYPE='q8_0')
            with (state / 'ollama.log').open('a') as log:
                if not adopted_identity:
                    server = popen([args.ollama, 'serve'], env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
                deadline = now() + args.start_timeout
                while not probe(args.endpoint):
                    if server is not None and server.poll() is not None:
                        raise RuntimeError('Ollama exited before becoming ready; see ollama.log')
                    if now() >= deadline:
                        raise RuntimeError('Ollama readiness timeout; see ollama.log')
                    sleep(args.poll_interval)
                if not args.adopt_existing and not existing:
                    result = run(['/usr/bin/open', '-n', '--env', 'CODEX_HOME=' + args.codex_home,
                                  '-a', 'ChatGPT', '--args', '--user-data-dir=' + args.profile],
                                 check=False)
                    if result.returncode:
                        raise RuntimeError('Codex Qwen app launch failed')
                deadline = now() + args.start_timeout
                while not qwen_pids(rows(), args.profile):
                    if args.adopt_existing:
                        app_closed = True
                        return 0
                    if now() >= deadline:
                        raise RuntimeError('Codex Qwen app did not stay open')
                    sleep(args.poll_interval)
                if args.ready_file:
                    Path(args.ready_file).write_text('ready')
                while qwen_pids(rows(), args.profile):
                    if server is not None and server.poll() is not None:
                        raise RuntimeError('Ollama exited while Codex Qwen remained open; see ollama.log')
                    sleep(args.poll_interval)
                app_closed = True
        finally:
            stop_group(server)
            if app_closed:
                for job in args.retire_job:
                    label, expected_pid = job.rsplit(':', 1)
                    target = 'gui/%s/%s' % (os.getuid(), label)
                    status = run(['/bin/launchctl', 'print', target], capture_output=True, text=True, check=False)
                    if status.returncode == 0 and ('pid = ' + expected_pid) in [line.strip() for line in status.stdout.splitlines()] :
                        run(['/bin/launchctl', 'bootout', target], check=True)
                if adopted_identity and next((command for pid, command in rows() if pid == args.adopt_server_pid), None) == adopted_identity:
                    os.kill(args.adopt_server_pid, signal.SIGTERM)
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['start', 'supervise'], help='start detaches supervisor; supervise runs foreground')
    p.add_argument('--adopt-existing', action='store_true', help='Monitor running Qwen app without reopening it')
    p.add_argument('--ready-file', help=argparse.SUPPRESS)
    p.add_argument('--adopt-server-pid', type=int, help='One-time adoption of exact existing patched server PID')
    p.add_argument('--retire-job', action='append', default=[], metavar='LABEL:PID', help='On app exit, boot out launchd job only if its PID matches')
    p.add_argument('--ollama', default=str(Path.home() / '.local/opt/ollama-qwen/bin/ollama'))
    p.add_argument('--endpoint', default='http://127.0.0.1:11435')
    p.add_argument('--profile', default=PROFILE)
    p.add_argument('--codex-home', default=str(Path.home() / '.codex/profiles/qwen'))
    p.add_argument('--state-dir', default=str(Path.home() / 'Library/Logs/Codex-Qwen'))
    p.add_argument('--start-timeout', type=float, default=30)
    p.add_argument('--poll-interval', type=float, default=1)
    return p


def main():
    args = parser().parse_args()
    if args.start_timeout <= 0 or args.poll_interval <= 0:
        parser().error('timeouts must be positive')
    if args.command == 'start':
        state = Path(args.state_dir)
        state.mkdir(parents=True, exist_ok=True)
        descriptor, ready_path = tempfile.mkstemp(prefix='startup-', dir=state)
        os.close(descriptor)
        with (state / 'supervisor.log').open('a') as log:
            command = [sys.executable, str(Path(__file__).resolve()), 'supervise', *sys.argv[2:], '--ready-file', ready_path]
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 2 * args.start_timeout + 3
        try:
            while time.monotonic() < deadline:
                if Path(ready_path).read_text() == 'ready':
                    print('Codex Qwen ready; log: %s' % (state / 'supervisor.log'))
                    return 0
                if child.poll() is not None:
                    print('Codex Qwen startup failed; see %s' % (state / 'supervisor.log'), file=sys.stderr)
                    return 1
                time.sleep(0.1)
            child.terminate()
            child.wait(timeout=12)
            print('Codex Qwen startup timed out; see %s' % (state / 'supervisor.log'), file=sys.stderr)
            return 1
        finally:
            Path(ready_path).unlink(missing_ok=True)
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        return supervise(args)
    except (RuntimeError, OSError) as error:
        print('Codex Qwen: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
