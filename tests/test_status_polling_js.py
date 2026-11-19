"""A delayed status request must not accumulate polls or block later recovery."""
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which('node') is None, reason='node is not installed')
def test_polling_serializes_requests_and_recovers_after_failure():
    page = (Path(__file__).resolve().parents[1] / 'src/templates/index.html').read_text()
    code = page[page.index('  let statusRequestInFlight'):page.index('  setInterval(pollStatus')]
    harness = r'''
let calls = 0, reject;
const fetch = () => { calls++; return new Promise((resolve, bad) => { reject = bad; }); };
const element = {replaceChildren() {}, style:{}};
const document = {createElement:() => ({})};
const messages=element, alertText=element, levelChip=element, alertBanner=element,
video=element, modelStatus=element, pipelineStatus=element, objectAlerts=element, ignoredList=element;
'''
    exercise = r'''
(async () => {
  const first = pollStatus();
  await pollStatus();
  if (calls !== 1) throw Error('Concurrent status request');
  reject(Error('Connection interrupted')); await first;
  const next = pollStatus();
  if (calls !== 2) throw Error('Polling failed to recover');
  reject(Error('Interrupted again')); await next;
  console.log('passed');
})().catch(e => {console.error(e); process.exitCode=1;});
'''
    result = subprocess.run(['node', '-e', harness + code + exercise],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert 'passed' in result.stdout
