import glob
import json
import os
import re
import subprocess
from datetime import datetime

GET_JOB_INFO = "sacct --format=jobid%20,jobname%100,state%30 --jobs="
run_folder_re = re.compile(r".*Run folder.*\'(.*)\'$")
run_id_re = re.compile(r".*ubmitted.*job (\d+)$")


def continue_job(jobid, logfile, slurm):
    outfile = logfile.replace("%j", str(jobid))
    outfile = re.sub(r"%[a-z,A-Z]", "*", outfile)
    files = glob.glob(outfile)
    if len(files) != 1:
        return -1
    file = files[0]
    with open(file, "r") as f:
        lines = f.readlines()
    runfolder = None
    for line in lines:
        match = run_folder_re.match(line)
        if match:
            runfolder = match.group(1)
            break
    if runfolder is None:
        return -1
    model_weights = sorted(
        [os.path.join(runfolder, f) for f in os.listdir(runfolder) if f.endswith(".pt")],
        key=lambda f: os.path.getmtime(f),
    )[-1]
    command = f"./main.py --task continue --model '{model_weights}' " + " ".join(
        f"--{key} {' '.join(val.split(',')) if key == 'partition' else val}" for key, val in slurm.items()
    )
    runout = subprocess.check_output(command, shell=True, text=True).split("\n")
    print(f"Continue job {jobid}")
    for line in runout:
        match = run_id_re.match(line)
        if match:
            return int(match.group(1))
    return -1


print(f"Run autocontinue at {datetime.now()}")
with open("autocontinue-jobs.jsonl", "r") as f:
    lines = f.readlines()

jobs = [json.loads(l.strip()) for l in lines if len(l.strip()) > 10]
ids = [job["jobid"] for job in jobs]
id_to_job = {j["jobid"]: j for j in jobs}

slurminfo = subprocess.check_output(GET_JOB_INFO + ",".join([str(i) for i in ids]), shell=True, text=True)
slurminfo = slurminfo.split("\n")

lenths = slurminfo[1].strip().split(" ")
id_len, name_len, state_len = [len(l) for l in lenths]
# print(id_len, name_len, state_len)

to_remove = {}
to_continue = {}
for line in slurminfo[2:]:
    if len(line) <= id_len + name_len:
        continue
    id = line[:id_len].strip()
    name = line[id_len + 1 : -state_len - 1].strip()
    state = line[-state_len:].strip()

    if "." in id:
        continue
    id = int(id)

    if "timeout" in state.lower():
        to_continue[id] = -1
    elif any([s in state.lower() for s in ["cancelled", "complete", "fail"]]):
        to_remove[id] = state


if len(to_remove) + len(to_continue) > 0:
    # print(f"continue: {to_continue}, remove: {to_remove}")
    continued_jobs = {}
    for jobid in to_continue:
        try:
            to_continue[jobid] = continue_job(**id_to_job[jobid])
        except Exception as e:
            print(f"Failed to continue job {jobid}: {e}")

    # removing jobs from file
    with open("autocontinue-jobs.jsonl", "r") as f:
        joblines = f.readlines()

    out_lines = []
    for line in joblines:
        line = line.strip()
        if len(line) < 10:
            continue
        jobinfo = json.loads(line.strip())
        jobid = jobinfo["jobid"]
        if jobid in to_remove.keys():
            print(f"Remove job {jobid} ({to_remove[jobid]})")
            continue
        if jobid in to_continue.keys():
            print(f"Job {jobid} continued as {to_continue[jobid]}")
            continue
        out_lines.append(json.dumps(jobinfo))
    with open("autocontinue-jobs.jsonl", "w") as f:
        for line in out_lines:
            f.write(line + "\n")
