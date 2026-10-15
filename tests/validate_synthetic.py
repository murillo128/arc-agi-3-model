"""On-demand synthetic acceptance: real SDK + all three real Chromium suites.

Fails on missing dependencies, wrong Python/SDK, test failures or mandatory
skips. This is intentionally not an always-on CI job. Large logs/screenshots
stay under ignored artifacts/. Optional originals must already be authorized
and locally cached; this command never downloads games.
"""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--originals-dir',type=Path)
    args=parser.parse_args()
    if sys.version_info[:2] != (3,12):
        parser.error('Required acceptance needs Python 3.12')
    dependencies={p:version(p) for p in ('arc-agi','arcengine','playwright','numpy')}
    if dependencies['arc-agi']!='0.9.9': parser.error('Requires arc-agi==0.9.9')
    output=ROOT/'artifacts'/'synthetic-validation'; output.mkdir(parents=True,exist_ok=True)
    metadata={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)),
              'python':sys.version,'dependencies':dependencies,
              'originals_dir':str(args.originals_dir) if args.originals_dir else None}
    (output/'environment.json').write_text(json.dumps(metadata,indent=2)+'\n')
    suite=unittest.TestSuite()
    for pattern in ('test_foundations.py','test_foundation_sdk.py','test_skill_curriculum.py','test_skill_sdk.py','test_square_cross*.py','test_play_game.py','test_symbolic_presentation.py'):
        suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=pattern))
    with (output/'rules-sdk-protocol.log').open('w') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    print(f'Rules/SDK/protocol: {result.testsRun} tests, {len(result.skipped)} skips',flush=True)
    if not result.wasSuccessful() or result.skipped:
        raise SystemExit(f'Required validation failed; see {output}/rules-sdk-protocol.log')
    with (output/'presentation-replay.log').open('w') as log:
        result=subprocess.run([sys.executable,str(ROOT/'tests/presentation_replay.py'),'--images'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode: raise SystemExit(f'Logical/pixel replay failed; see {output}/presentation-replay.log')
    print('Presentation replay: PASS',flush=True)
    for name in ('play_game_browser','skill_browser','foundation_browser'):
        command=[sys.executable,'-u',str(ROOT/'tests'/f'{name}.py')]
        if args.originals_dir: command+=['--originals-dir',str(args.originals_dir.resolve())]
        env=dict(os.environ,PYTHONPATH=str(ROOT))
        with (output/f'{name}.log').open('w') as log:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(f'Required browser validation failed; see {output}/{name}.log')
        print(f'{name}: PASS',flush=True)
    print(f'PASS: required synthetic acceptance; evidence in {output}',flush=True)


if __name__=='__main__': main()
