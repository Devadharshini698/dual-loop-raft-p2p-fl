# Physical Testbed and Judge Demonstration Guide

## Purpose and claim boundary

The repository now has two execution paths:

1. The original deterministic `VirtualClock` and `SimNetwork` path for repeatable
   unit, property and RQ experiments.
2. The optional `physical_testbed` path using monotonic wall time, real UDP Raft
   messages, persistent boot/Raft state, deployed Flower apps and JSONL evidence.

`python run_physical_demo.py` is a single-computer real-socket smoke test. It does
not establish a multi-machine physical result. A physical claim begins only after
the adapter runs on separate laboratory computers and its evidence passes validation.

## Laboratory topology

Use one wired controller and five wired peers on an isolated switch or VLAN.
The controller hosts the Observatory, evidence collection, Flower SuperLink and
ServerApp. Each peer hosts one Raft process, one Flower SuperNode and the Flower
ClientApp. Do not use the experimental interface as the only SSH path.

| Host | Suggested address | Role |
|---|---:|---|
| Controller | `10.20.0.10` | SuperLink, ServerApp, orchestration, evidence |
| node1 | `10.20.0.11` | Raft peer and Flower partition 0 |
| node2 | `10.20.0.12` | Raft peer and Flower partition 1 |
| node3 | `10.20.0.13` | Raft peer and Flower partition 2 |
| node4 | `10.20.0.14` | Raft peer and Flower partition 3 |
| node5 | `10.20.0.15` | Raft peer and Flower partition 4 |

## 1. Pre-lab verification

On the development computer:

```bash
python -m pytest -q
python run_physical_demo.py
flwr build --app flowerapp
```

On Windows PowerShell, use
`$env:PYTHONUTF8='1'; flwr build --app flowerapp` if the console reports a
CP-1252 `UnicodeEncodeError` after writing the `.fab`.

The smoke result is written to `results/physical_smoke.json`. The Flower build
must produce a `.fab` without configuration errors.

## 2. Prepare the machines

Copy or clone the same Git commit to `/opt/dual-loop` on all six machines. On
each machine run:

```bash
bash physical_testbed/scripts/bootstrap_host.sh /opt/dual-loop
```

Copy `physical_testbed/configs/lab.example.json` to `/etc/dual-loop/lab.json`
and replace every address, interface and SSH user with the laboratory values.
Verify connectivity before introducing any fault:

```bash
ping -c 20 10.20.0.12
chronyc tracking
python -m pytest -q
```

All machines must report Flower `1.40.0`, use the same Git commit, and have
synchronized clocks.

Create the controller-side manifest before starting the experiment:

```bash
python -m physical_testbed.manifest \
  --config /etc/dual-loop/lab.json --run-id judge-demo-001 \
  --output /var/log/dual-loop/runs/judge-demo-001/manifest.json
```

## 3. Start the five Raft peers

For a manual first run, execute the matching command on every peer:

```bash
python -m physical_testbed.node \
  --config /etc/dual-loop/lab.json \
  --node-id node1 \
  --run-id judge-demo-001
```

For managed restart testing, install `physical_testbed/systemd/dual-loop-node@.service`,
create `/etc/dual-loop/runtime.env`, and start the matching instance:

```bash
sudo install -D -m 0644 physical_testbed/systemd/dual-loop-node@.service \
  /etc/systemd/system/dual-loop-node@.service
sudo install -D -m 0644 physical_testbed/systemd/runtime.env.example \
  /etc/dual-loop/runtime.env
sudo systemctl daemon-reload
sudo systemctl enable --now dual-loop-node@node1
```

Every peer publishes `status-nodeN.json` and appends `nodeN.jsonl`. Restarts
increment a durable `boot_id`; terms, votes, log entries and commit index are
stored atomically before dependent protocol messages are sent.
Real heartbeat silence starts a frozen suspicion, witness replies or a returning
boot identifier resolve it, and resolved non-ambiguous labels train the same
telemetry extractor and fast loop used by the deterministic experiments.

Capture packet and process evidence on each peer during a measured interval:

```bash
bash physical_testbed/scripts/capture_evidence.sh \
  eno2 "$(systemctl show -p MainPID --value dual-loop-node@node1)" 60 \
  /var/log/dual-loop/runs/judge-demo-001/node1-evidence
```

## 4. Start deployed Flower

On the controller:

```bash
bash physical_testbed/scripts/start_superlink.sh
bash physical_testbed/scripts/configure_flower_cli.sh 10.20.0.10
```

On each peer, use its partition, node ID and current Raft status path:

```bash
bash physical_testbed/scripts/start_supernode.sh \
  10.20.0.10 0 node1 \
  /var/log/dual-loop/runs/judge-demo-001/status-node1.json
```

After all five SuperNodes connect, submit the Flower app from the controller:

```bash
bash physical_testbed/scripts/run_flower_experiment.sh
```

The `RaftGatedFedAvg` strategy accepts updates only when at least three peers
report the same Raft `(term, leader)` pair. Stale and minority-term updates are
rejected. SuperLink remains the Flower transport/control plane; Raft remains the
logical authority gate. The supplied `--insecure` profile is for an isolated lab
only. Use Flower TLS and SuperNode authentication outside that environment.

## 5. Fault demonstrations

All fault commands are dry runs unless `--apply` is present. The controller logs
both the planned command and whether it was executed.

Transient link outage:

```bash
python -m physical_testbed.faultctl \
  --config /etc/dual-loop/lab.json --run-id judge-demo-001 \
  --node-id node3 --apply loss --percent 100 --duration 0.6 --seed 42
```

Healthy-slow link:

```bash
python -m physical_testbed.faultctl \
  --config /etc/dual-loop/lab.json --run-id judge-demo-001 \
  --node-id node4 --apply delay --milliseconds 100 --jitter 30 \
  --duration 20 --seed 42
```

Crash and recovery:

```bash
python -m physical_testbed.faultctl \
  --config /etc/dual-loop/lab.json --run-id judge-demo-001 \
  --node-id node2 --apply crash --duration 3
```

The controller always attempts cleanup in a `finally` block. Nevertheless, keep
physical-console access and verify cleanup with `tc -s qdisc show dev eno2`.
Never apply `netem` to the general college network or the only management link.

## 6. Required experiment order

Run clean baseline, 600 ms transient link failure, asymmetric loss, healthy-slow,
crash/recovery, permanent crash, observer isolation, Regime A-to-B drift, leader
death during FL, poisoned encoder, and cold-start join. For paper results, pair
every method with the identical seeded schedule and repeat each main condition 30
times. For a judge demo, rehearse three successful runs of selected scenarios.

## 7. Evidence and acceptance

Each run must contain:

```text
runs/<run-id>/
  fault-events.jsonl
  node1.jsonl ... node5.jsonl
  status-node1.json ... status-node5.json
  flower-rounds.csv
  resource-metrics.csv
  network.pcap
  manifest.json
```

Validate the run from the controller:

```bash
python -m physical_testbed.collect \
  --config /etc/dual-loop/lab.json --run-id judge-demo-001 \
  --output-root /var/log/dual-loop/collected --apply
python -m physical_testbed.validate_evidence \
  /var/log/dual-loop/collected/judge-demo-001
```

Acceptance requires one leader per term, monotonic terms, consistent committed-log
checksums, a ground-truth fault ledger, no timeout below the safety floor, and no
stale-term Flower update accepted. Report label noise together with unresolved
coverage, Macro-F1, per-class recall, drift recovery time, elections per hour,
FL time to 80% accuracy, stalls, handovers, CPU, RSS and network bytes.

## Judge demonstration sequence

1. Show six labelled machines and the Git commit on each.
2. Show the five peers agreeing on one leader and term.
3. Start a Flower round and show the accepted consensus signature.
4. Inject the 600 ms link outage; show same boot ID and link-failure handling.
5. Crash the leader; show boot-ID change, safe election and FL continuation.
6. Show `fault-events.jsonl`, node evidence and the validation PASS report.
7. State explicitly which view is deterministic simulation and which evidence is
   from separate physical hosts.
