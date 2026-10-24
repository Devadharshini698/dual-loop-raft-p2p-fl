# Deterministic RQ Evaluation Summary

Synthetic seeded benchmarks for implementation validation. Results are not deployment measurements. CPU timing is machine-dependent and can vary between runs.

## RQ1

| Variant | Metric | Value | Unit |
|---|---|---:|---|
| DualLoop | macro_f1 | 1 | ratio |
| OracleOfflineML | macro_f1 | 1 | ratio |
| DualLoop | label_noise_rate | 0 | ratio |
| DualLoop | resolution_coverage | 0.948889 | ratio |

## RQ2

| Variant | Metric | Value | Unit |
|---|---|---:|---|
| FixedTimeout | recovery_samples | 118 | samples |
| FixedTimeout | spurious_elections_per_hour | 9303.64 | events_per_hour |
| PhiAccrual | recovery_samples | 100 | samples |
| PhiAccrual | spurious_elections_per_hour | 646.086 | events_per_hour |
| SWIMLifeguard | recovery_samples | 0 | samples |
| SWIMLifeguard | spurious_elections_per_hour | 0 | events_per_hour |
| BanditTimeoutTuner | recovery_samples | 2 | samples |
| BanditTimeoutTuner | spurious_elections_per_hour | 258.434 | events_per_hour |
| OfflineStaticML | recovery_samples | 180 | samples |
| OfflineStaticML | spurious_elections_per_hour | 11148.9 | events_per_hour |
| DualLoop | recovery_samples | 117 | samples |
| DualLoop | spurious_elections_per_hour | 510.638 | events_per_hour |

## RQ3

| Variant | Metric | Value | Unit |
|---|---|---:|---|
| FastLoopOnly | rare_crash_recall | 0.985075 | ratio |
| FastLoopOnly | cold_start_macro_f1 | 0.527614 | ratio |
| SlowLoopOnly | rare_crash_recall | 0.985075 | ratio |
| SlowLoopOnly | cold_start_macro_f1 | 0.95044 | ratio |
| DualLoop_NoDriftWeighting | rare_crash_recall | 0 | ratio |
| DualLoop_NoDriftWeighting | cold_start_macro_f1 | 0.355035 | ratio |
| FullDualLoop | rare_crash_recall | 1 | ratio |
| FullDualLoop | cold_start_macro_f1 | 0.982436 | ratio |

## RQ4

| Variant | Metric | Value | Unit |
|---|---|---:|---|
| P2PFL | cpu_time_per_step | 0.792475 | ms |
| P2PFL | model_memory | 168 | bytes |
| P2PFL | message_overhead | 7560 | bytes |
| P2PFL | stalled_rounds | 1 | rounds |
| P2PFL | time_to_target_accuracy | 2000 | virtual_ms |
| P2PFL | final_accuracy | 1 | ratio |

## RQ5

| Variant | Metric | Value | Unit |
|---|---|---:|---|
| FixedTimeout + ProvisionalLeader | stall_duration | 225 | ms |
| FixedTimeout + ProvisionalLeader | spurious_handovers | 20 | count |
| DualLoop + GatedProvisionalLeader | stall_duration | 135 | ms |
| DualLoop + GatedProvisionalLeader | spurious_handovers | 0 | count |
