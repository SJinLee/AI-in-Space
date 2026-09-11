# SEU suite — LEO bit-flip simulation / LEO 비트플립 시뮬레이션

Reusable multi-model suite for **true single-event upsets** (SEU) on neural-network weights.
진짜 SEU = float32 가중치의 **하나의 비트를 XOR**. 가우시안 노이즈가 아닙니다.

Work directory: `seu_suite/`.  Do **not** delete `exp1_*` / `train_leo_*` in the parent folder.

---

## 0. One-cell re-run (학생용 / for students)

From this folder, with the CPU venv:

```bash
cd /workspace/AI-in-Space-experiment/seu_suite
/workspace/seu-venv/bin/python simulate.py --model toy50 --mode infer --rad leo_saa --defense clip --trials 20
```

That is one “cell”: one model × one mode × one radiation × one defense × N trials.
Results go to `results/single_toy50_infer_leo_saa_clip.csv`.

Full paper matrix (writes combined CSVs + figures):

```bash
/workspace/seu-venv/bin/python simulate.py --preset paper_sweep
```

---

## 1. What a true SEU is / 진짜 SEU란

| 한국어 | English |
|--------|---------|
| 가중치 하나를 IEEE-754 float32로 보고, `uint32` 뷰에서 **비트 하나만 XOR** | View one weight as float32, XOR **exactly one bit** of its uint32 representation |
| 가우시안 노이즈 / `weight += σ·N(0,1)` 가 **아님** | This is **not** Gaussian weight noise |
| 기본 비율 `r = 1e-6` SEU/bit/day (`--r`) | Default rate `r = 1e-6` SEU/bit/day (`--r`) |
| `λ = r × N_bits × Δt_days` | Expected count is a Poisson mean `λ = r * N_bits * Δt_days` |
| `dim≥2` 인 **weight 텐서만** 플립 (bias 제외, `--flip-bias` 로 포함) | Only tensors with `dim≥2` (matrices / conv kernels). Biases only with `--flip-bias` |
| `N_bits` 는 로그와 CSV에 기록됨 | `N_bits` is printed and stored |

Self-test: XOR of the sign bit of `1.0` must yield `-1.0`.

---

## 2. Radiation modes (`--rad`)

### `poisson` (simple)
Homogeneous Poisson process on the experiment window `[0, T]`.
Sample `K ~ Poisson(λ)`, drop event times uniform in `[0, T]`.
Each event = `(weight_index, bit 0–31)`.

### `leo_saa`
One **mission day** (or a scaled window of length `T`).
Geometry of a 1-day LEO mission:

- 3 South-Atlantic-Anomaly (SAA) passes × **12 min**
- Put **~90%** of expected daily SEUs into those passes
- `r_saa ≈ r_avg * 0.90 / (3*12/1440) ≈ 36 r_avg`
- Remainder in quiet time (`r_quiet ≈ 0.1026 r_avg`)
- Pass start times: equally spaced slots with uniform jitter (non-overlapping)
- For `T ≠ 1` day, `n_passes = max(1, round(3·T))` (so `--T saa_pass` → one 12-minute pass)

### `stress` (course-style accelerator)
**Ignores physical `r` and `T`.**
Hit each weight independently with probability `p` (default `--p 0.01`; also try `0.05`),
then flip one random bit. This is the week-13/15 classroom accelerator, **not** a LEO fluence.

---

## 3. Time `T` — read this before quoting numbers

**Inference.** `T` is a chosen exposure.

| flag | meaning |
|------|---------|
| `--T 1.0` (default) | one LEO day |
| `--T saa_pass` | one SAA pass = `12/1440` days |

**Training.** We do **not** use Colab wall-clock as satellite time.

Two options were considered:

- **(a) (default in this suite)** assume a satellite-equivalent exposure via `--train-hours` (default **24**), mapping the **whole training run** to `T = train_hours/24` days so `λ` is nontrivial for `leo_saa` / `poisson`.
- (b) per-step Poisson with `dt = train_hours / n_steps`.

This suite uses **(a)** with `--train-hours 24` for leo/poisson: the scenario is **“one day of training”**.

> **ASSUMPTION, not wall-clock.** A 2-second training run on a laptop is **not** “a satellite trained for 2 seconds in LEO”. `--train-hours 24` *declares* that the run is equivalent to 24 h of on-orbit exposure. Document this on any poster.

Stress mode does not use `T`.

Event times in `[0, T]` map onto optimizer steps: `step = floor((t/T) * n_steps)`.
An event at 30% of the window is injected at the 30% step. Defense runs after injection.

If training hits NaN/Inf loss (or logits/grads/weights), `crashed=1` and we record the last finite accuracy, else chance-level `1/n_classes`.

---

## 4. Defenses (`--defense`)

| name | inference | training |
|------|-----------|----------|
| `none` | leave flipped bits | same |
| `clip` | `nan_to_num` + clamp **all** params to `[-3, 3]` after injection **and** before eval | after each injection, and before eval |
| `tmr` | 3 independently corrupted copies of clean `W`, **per-weight median** | after each injection: 3 independent corruptions of *pre-injection* `W` (same `K`, independent locations), median → live `W`. Expensive; OK on small models |
| `ensemble` | 5 independently corrupted models, **average logits** | 3-model logit-average **only** on `toy50` / `mnist_linear`. CNN / ResNet: **inference-only** (documented) |
| `parity` | even parity of each weight’s 32 bits stored alongside; on mismatch, **zero** that weight | on mismatch, **revert** that weight to the pre-flip snapshot; refresh parity after a legitimate optimizer step |

---

## 5. Models (`--model`)

| name | what | metric |
|------|------|--------|
| `toy50` | 50 float32 weights as `Linear(50,1,bias=False)` (dim≥2 so it is flippable). Course toy. | **infer:** sign-agreement vs clean on `N(0,1)` inputs. **train:** short SGD on a synthetic teacher-sign task (not MNIST) |
| `mnist_linear` | Flatten 784 → 128 → 10 | MNIST top-1 % |
| `mnist_cnn` | Conv32–Conv64–FC, 28×28 | MNIST top-1 % |
| `resnet18_t10` | torchvision ResNet-18, 10-class, **CIFAR-10 32×32 (no 224 upscale)**. Prefer ImageNet pretrained, freeze backbone, train last layer. If download fails: reduced ResNet-18 (`conv1` 3×3 stride 1, no maxpool) from scratch on a CIFAR subset — still called `resnet18_t10` | CIFAR-10 top-1 % |
| `cifar_deep` | 3 conv stages, ~300k params | CIFAR-10 top-1 % |

`N_bits` is counted from weight tensors only and printed in the logs.

---

## 6. CLI flags

| flag | default | meaning |
|------|---------|---------|
| `--model` | `toy50` | one of the five names above |
| `--mode` | `infer` | `infer` (irradiate a cached clean checkpoint) or `train` (inject during SGD) |
| `--rad` | `leo_saa` | `poisson` / `leo_saa` / `stress` |
| `--defense` | `none` | `none` / `clip` / `tmr` / `ensemble` / `parity` |
| `--trials` | `5` | independent repeats; each has a recorded seed |
| `--r` | `1e-6` | SEU/bit/day (ignored by `stress`) |
| `--T` | `1.0` | exposure in days, or `saa_pass` |
| `--p` | `0.01` | stress per-weight hit probability |
| `--train-hours` | `24` | **ASSUMPTION**: map the training run to this many hours of satellite time |
| `--flip-bias` | off | also flip `dim==1` tensors |
| `--epochs` / `--n-train` / `--n-test` | model default | override budget |
| `--preset paper_sweep` | — | run the CPU-feasible matrix, write combined CSV + figures |
| `--out-dir` | `seu_suite/results` | CSV / PNG / NOTES |
| `--seed-master` | `2026` | master seed; every trial still gets its own recorded seed |

---

## 7. Outputs of `--preset paper_sweep`

- `results/sweep_raw.csv` — one row per trial
- `results/sweep_summary.csv` — groupby `model, mode, rad, defense`: n, mean_acc, std, crash_rate, mean_k, n_bits
- `results/acc_infer_leo_saa_by_defense.png`
- `results/acc_stress_vs_leo.png`
- `results/crash_rate.png`
- `results/NOTES.md` — actual `N_bits`, T assumptions, what ran, timings
- `results/ckpts/*_clean.pt` — cached clean weights for infer

---

## 8. Environment

```bash
python3 -m venv /workspace/seu-venv
/workspace/seu-venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
/workspace/seu-venv/bin/pip install numpy pandas matplotlib
```

No GPU. MNIST / CIFAR-10 download into `/workspace/data` on first use.

---

## 9. Physics reminder for posters

Do **not** claim `leo_saa` on a 2-second Colab train is “real satellite wall-clock”.
The suite uses `--train-hours 24` as the physical exposure **assumption**.


---

## 10. Changelog 2026-09-11 (review fixes) / 점검 후 수정 사항

| # | 무엇을 | 왜 |
|---|---|---|
| 1 | `leo_saa`: 통과 횟수를 `round(3T)` 대신 **정수부 + Bernoulli(소수부)** 로 샘플링 | T가 하루의 정수배가 아니면 선량이 r·N_bits·T에 비례하지 않았음(T=0.5 → 1.3배). 이제 모든 T에서 E[λ] = r·N_bits·T. `--T saa_pass`(한 번 통과 = 하루 선량의 30%) 의미는 유지 |
| 2 | 새 방어 `ensemble_median` | 기존 `ensemble`(로짓 평균)은 지수 최상위 비트가 뒤집혀 10^38 크기의 **유한한** 로짓을 내는 복제본 하나에 무너짐(복제본 4개가 100%인데 앙상블 64%). 로짓 중앙값은 이에 강건. `ensemble`은 "순진한 평균" 대조군으로 유지 |
| 3 | `radiation._u32_view`: CPU·float32가 아니면 예외 | CUDA 텐서에서는 `.cpu().numpy()`가 복사본이라 플립이 조용히 사라짐(Colab GPU에서 가짜 "무피해" 결과) |
| 4 | 모든 `except Exception`에 `traceback.print_exc()` + CSV `note` 열에 예외 기록 | 코딩 오류가 "붕괴(crash)"로 둔갑하지 않도록 |
| 5 | `trial_seed` 셀 간 간격 20 → 1000 | 반복 20회 초과 시 인접 셀과 시드 충돌 가능. **8/19 파일럿과 시드가 달라짐**(파일럿은 CSV의 seed 열로 재현 가능) |
| 6 | 체크포인트 파일명에 `n{n_train}_e{epochs}` 포함 | `--n-train`/epochs를 바꿔도 예전 파일을 재사용하던 문제 |
| 7 | 추론 행 `epochs` 열 = 체크포인트 학습 epoch; `clean_acc` = 시행과 같은 지표로 잰 값(toy50은 자기 자신과의 부호 일치 = 100) | 메타데이터 오기, toy50 상대 정확도 > 1 문제 |
| 8 | `--allowed-bits`, `--single-hit` 옵션 (Cell 필드, CSV 열 포함) | 비트 위치 치명도 실험(E2), 영역 제한 실험(E6c) |
| 9 | 요약 CSV에 `median_acc`, `iqr_acc`, `survival_rate`(붕괴 없음 ∧ acc ≥ 0.9·clean), `p` 열 | 결과가 이봉 분포라 평균±표준편차만으로는 결론 불가 |
| 10 | 단일 실행 CSV 이름에 p/T/r/bits 태그, `--tag` | p를 바꿔 여러 번 돌릴 때 덮어쓰기 방지 |
| 11 | `boxplot` matplotlib 3.9 미만 호환, `write_notes` 잔재 정리 | |

### 알아둘 것 (코드 동작은 그대로)
- **stress 학습 선량**: 수업 코드는 매 epoch 끝마다 확률 p로 플립(8 epoch = 8회 선량), 이 스위트는 총 p·N_w번을 전체 스텝에 분산(1회 선량). 논문에 명시.
- **TMR 학습**: 샘플된 이벤트의 위치·비트를 쓰지 않고 복제본마다 k개의 새 무작위 플립을 줌. TMR 학습 행의 `n_exponent_hits`/`n_sign_hits`는 실제 적용과 무관.
- **앙상블 추론**: 복제본 5개가 각각 전체 선량을 받음(시스템 총 명중 5배). `k` 열은 복제본 평균.
- **붕괴 정의**: 가중치·로짓·손실·그래디언트 어디든 NaN/∞ → 시행 전체를 찍기 수준(10% / 50%)으로 기록. 수업 코드(샘플별 오답 처리)와 다름.

### 새 옵션 예시
```bash
# 비트 30번 하나만, 정확히 한 번 뒤집기 (E2 비트 치명도)
python simulate.py --model mnist_linear --mode infer --rad stress --single-hit --allowed-bits 30 --trials 50
# 지수부 전체 / 가수부 전체 / 부호만
python simulate.py --model mnist_cnn --mode infer --rad stress --p 0.001 --allowed-bits exponent --trials 10
# 강건한 앙상블
python simulate.py --model toy50 --mode infer --rad stress --p 0.05 --defense ensemble_median --trials 20
```
