#!/usr/bin/env node
'use strict';

const { spawnSync } = require('child_process');
const os = require('os');

const args = process.argv.slice(2);
const isWin = os.platform() === 'win32';
const installSpec = process.env.ROBOTMEM_PIP_SPEC || 'robotmem';
const pythonLaunchers = isWin
  ? [
      { command: 'py', args: ['-3'] },
      { command: 'py', args: [] },
      { command: 'python', args: [] },
      { command: 'python3', args: [] },
    ]
  : [
      { command: 'python3', args: [] },
      { command: 'python', args: [] },
    ];

// --- helpers ---

function run(cmd, cmdArgs, opts) {
  return spawnSync(cmd, cmdArgs, { stdio: 'inherit', shell: false, ...opts });
}

function runQuiet(cmd, cmdArgs, opts) {
  return spawnSync(cmd, cmdArgs, { stdio: 'pipe', shell: false, ...opts });
}

function hasCmd(cmd) {
  const r = runQuiet(isWin ? 'where' : 'which', [cmd]);
  return r.status === 0;
}

function fail(msg) {
  process.stderr.write(msg + '\n');
  process.exit(1);
}

function findPython() {
  if (process.env.ROBOTMEM_PYTHON) {
    return { command: process.env.ROBOTMEM_PYTHON, args: [] };
  }

  for (const launcher of pythonLaunchers) {
    if (!hasCmd(launcher.command)) continue;
    const check = runQuiet(launcher.command, [...launcher.args, '--version']);
    if (check.status === 0) return launcher;
  }

  return null;
}

function canImportRobotmem(python) {
  const check = runQuiet(python.command, [...python.args, '-c', 'import robotmem']);
  return check.status === 0;
}

function hasPip(python) {
  const check = runQuiet(python.command, [...python.args, '-m', 'pip', '--version']);
  return check.status === 0;
}

function runRobotmem(python) {
  return run(python.command, [...python.args, '-m', 'robotmem', ...args]);
}

// --- main ---

const python = findPython();

if (!python) {
  fail(
    'No supported Python interpreter found in PATH.\n' +
    'Install Python 3.10+ and ensure `python`, `python3`, or `py` is available.'
  );
}

// 1. Already installed? Forward directly.
if (canImportRobotmem(python)) {
  const r = runRobotmem(python);
  process.exit(r.status || 0);
}

// 2. Not installed. Try auto-install.
process.stderr.write(`robotmem not found, installing ${installSpec} for the current user...\n`);

if (!hasPip(python)) {
  fail(
    'pip is not available for the selected Python interpreter.\n' +
    `Run manually: ${python.command} ${python.args.join(' ')} -m ensurepip --upgrade`
  );
}

const install = run(python.command, [
  ...python.args,
  '-m',
  'pip',
  'install',
  '--user',
  '--upgrade',
  installSpec,
]);

if (install.status !== 0) {
  fail(`Automatic install failed. Run manually: ${python.command} ${python.args.join(' ')} -m pip install --user --upgrade ${installSpec}`);
}

// 3. Run the command after installation.
if (canImportRobotmem(python)) {
  const r = runRobotmem(python);
  process.exit(r.status || 0);
}

fail(
  'Installed successfully, but `python -m robotmem` still failed to import.\n' +
  'Check Python user-site settings or rerun with ROBOTMEM_PYTHON pointing to the intended interpreter.'
);
