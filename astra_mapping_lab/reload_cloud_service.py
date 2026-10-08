"""Run on server: reload only this idle cloud service, preserving its environment."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
from urllib.request import urlopen


def main():
    lab=Path(__file__).resolve().parent;server=lab/'cloud_server.py';jobs=lab/'jobs'
    active=[]
    for file in jobs.glob('*/state.json'):
        s=json.loads(file.read_text())
        if s['state'] not in ('UPLOADING','READY','FAILED','INTERRUPTED'):active.append((file.parent.name,s['state']))
    if active:raise RuntimeError('Refusing reload while jobs are active: '+repr(active))
    matches=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:args=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
        except (OSError,UnicodeError):continue
        # Home may be a symlink to the data disk. Compare canonical script
        # paths, while retaining the exact port check and original argv.
        same_server=any(Path(arg).is_absolute() and Path(arg).name==server.name
                        and Path(arg).resolve()==server for arg in args)
        if same_server and '--port' in args and args.index('--port')+1<len(args) and args[args.index('--port')+1]=='8030':matches.append((int(proc.name),args))
    if len(matches)!=1:raise RuntimeError('Expected exactly one matching own cloud service')
    pid,args=matches[0]
    env=dict(item.decode().split('=',1) for item in (Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0') if b'=' in item)
    # No environment values or capability tokens are printed or written.
    subprocess.run([args[0],'-m','compileall','-q',str(lab)],check=True,env=env)
    os.kill(pid,signal.SIGTERM)
    for _ in range(100):
        try:state=(Path('/proc')/str(pid)/'stat').read_text().split()[2]
        except FileNotFoundError:break
        if state=='Z':break
        time.sleep(.1)
    else:raise RuntimeError('Cloud process did not exit; refusing forced termination')
    with (lab/'cloud_service.log').open('ab') as log:
        child=subprocess.Popen(args,env=env,cwd=lab,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(40):
        try:
            with urlopen('http://127.0.0.1:8030/health',timeout=1) as r:d=json.load(r)
            from automatic_update import PROFILE
            if d.get('pipeline_profile')==PROFILE:
                print(json.dumps(dict(reloaded=True,pid=child.pid,health=d)));return
        except OSError:pass
        time.sleep(.25)
    raise RuntimeError('New cloud service did not pass health check; inspect cloud_service.log')

if __name__=='__main__':main()
