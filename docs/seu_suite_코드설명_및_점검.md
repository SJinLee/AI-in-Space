# seu_suite 코드 설명 및 점검 보고

> 대상: `sim/seu_suite.zip` (2026-08-19 버전, 6개 모듈 약 2,300줄)
> 점검일: 2026-09-11 · 방법: 전체 코드 정독 + 격리 가상환경(torch 2.14 CPU, numpy 2.5, matplotlib 3.11)에서 단위 점검 20항목과 실제 실행 16셀
> 결론: **치명적 버그 없음. 실행 오류 없음.** 논문 실험 전에 고쳐야 할 "의미상 함정" 4개와 개선 권고 8개를 아래에 정리.
>
> **2026-09-11 수정 완료.** 아래 2-3절 ①~④와 2-4절 5·6·7·12·13, 2-5절의 `--allowed-bits`·`--single-hit`·생존률 열을 모두 반영한 수정본이 `sim/seu_suite/`(압축 해제 폴더)에 있다. 원본 `seu_suite.zip`은 그대로 두었다. 변경 목록은 `seu_suite/README.md` 10절, 수정 후 재검증은 아래 2-6절.

---

## 1. 전체 구조

```
seu_suite/
├─ simulate.py   CLI 진입점. 단일 셀 실행 / paper_sweep 프리셋 / 요약 CSV·그림·NOTES 작성   (732줄)
├─ presets.py    paper_sweep 셀 목록(모델×모드×방사선×방어×반복)과 파일럿 후 축소 규칙
├─ engine.py     한 번의 시행(trial)을 실제로 돌리는 곳: 추론 시행, 학습 시행, 붕괴 판정, 클린 체크포인트
├─ radiation.py  물리: 이벤트(시각, 가중치 번호, 비트 번호) 샘플링과 float32 한 비트 XOR 주입
├─ defenses.py   clip / tmr / ensemble / parity 구현
├─ models.py     5개 모델(toy50, mnist_linear, mnist_cnn, cifar_deep, resnet18_t10), 데이터 로더, 기본 예산
└─ results/      파일럿 산출물 (sweep_raw.csv 670행, sweep_summary.csv, 그림 3장, NOTES.md)
```

### 1-1. 한 시행이 흘러가는 순서 (추론 모드)

1. `simulate.run_cell` → 데이터 로드(캐시) → `load_or_train_clean`으로 깨끗한 체크포인트 확보(없으면 학습해서 `results/ckpts/<모델>_clean.pt` 저장)
2. `engine.run_infer_trial` → 시행별 시드로 `np.random.default_rng`, `torch.manual_seed`
3. `_one_irradiated`: 새 모델에 클린 가중치 로드 → `collect_weight_params`(dim ≥ 2 텐서만) → `weight_bank_meta`(텐서별 누적 개수, 전역 번호→(텐서, 위치) 변환용) → `sample_schedule`로 이벤트 목록 생성 → `apply_events`가 이벤트마다 `xor_one_bit` 호출
4. `xor_one_bit`: 텐서를 `numpy().view(np.uint32)`로 같은 메모리를 정수로 보고 `^= 1 << bit`. CPU에서는 numpy 배열이 텐서 메모리를 공유하므로 모델 가중치가 **제자리에서** 바뀜 (9주차 함수와 같은 원리)
5. 방어 적용(clip → `nan_to_num` + `clamp(-3,3)` / parity → 패리티 불일치 가중치 0으로 / tmr·ensemble은 복제본 3·5개를 각각 3번 만든 뒤 중앙값·로짓 평균)
6. `evaluate`: 가중치나 출력에 NaN/∞가 하나라도 있으면 `None` → 붕괴(crashed=1), 정확도는 찍기 수준(10% 또는 50%)으로 기록

### 1-2. 학습 모드가 다른 점

- 이벤트를 학습 시작 전에 한꺼번에 샘플링하고, `events_by_step`이 시각 t∈[0,T]를 옵티마이저 스텝 번호로 바꿈 (`step = floor(t/T × 총스텝)`)
- 매 스텝: 해당 스텝의 이벤트가 있으면 (스냅샷 → XOR 주입 → 방어) → 가중치·로짓·손실·그래디언트·갱신 후 가중치 5단계 유한성 검사 → 하나라도 NaN/∞면 `crashed=1`로 중단하고 마지막 유한 정확도를 기록
- 비-stress 모드는 `--train-hours 24` 가정으로 "학습 전체 = 궤도 1일"(T=1.0). stress 모드는 T 무시(총 p·N번을 스텝에 고르게 분산)
- parity 학습: 옵티마이저 스텝마다 패리티 표를 새로 계산, 주입 직후 불일치 가중치를 주입 전 스냅샷으로 되돌림
- tmr 학습: 주입 전 스냅샷에서 복제본 3개를 만들어 각각 k개의 **독립 무작위** 플립을 주고 원소별 중앙값을 살아있는 가중치로 씀

### 1-3. 방사선 3모드 (radiation.py)

| 모드 | 이벤트 수 | 시각 분포 | 비고 |
|---|---|---|---|
| `poisson` | K ~ Poisson(r·N_bits·T) | [0,T] 균등 | 가장 단순한 물리 모델 |
| `leo_saa` | SAA 구간 K ~ Poisson(36r·N_bits·SAA시간), 조용한 구간 K ~ Poisson(0.1026r·N_bits·나머지) | 하루 3회 × 12분 SAA 통과(슬롯별 무작위 지터, 비중첩) | T=1일이면 총 λ = r·N_bits (검증됨), 이벤트의 90%가 SAA 안 (검증: 90.1%) |
| `stress` | 각 가중치 독립 확률 p | [0,T] 균등 | 수업 방식. λ 대신 기대치 p·N_w |

비트 번호는 0~31 균등. `bit_kind`: 31=부호, 23~30=지수, 0~22=가수. 한 가중치가 두 번 맞을 수 있음(물리적으로 타당).

### 1-4. 방어 4종 (defenses.py)

| 방어 | 추론 | 학습 | 비용 |
|---|---|---|---|
| clip | `nan_to_num` 후 **모든 파라미터**(편향·BN 포함)를 [−3, 3]으로 | 주입 직후 + 평가 전 | ≈0 |
| tmr | 클린 가중치의 독립 오염 복제본 3개 → 원소별 중앙값 (∞는 0으로 바꾼 뒤 중앙값) | 위 1-2 | ×3 |
| ensemble | 독립 오염 복제본 5개 → 로짓 `nan_to_num` 후 평균 | toy50·mnist_linear만 복제본 3개 학습, CNN/ResNet은 none으로 대체하고 `ensemble_skipped=1` 표시 | ×5 |
| parity | 32비트 짝수 패리티(SWAR popcount) 저장, 불일치 가중치 0 | 불일치 가중치를 주입 전 값으로 복원 | +1/32 메모리 |

### 1-5. 모델과 지표 (models.py)

| 모델 | 구조 | 가중치 | 지표 |
|---|---|---|---|
| toy50 | Linear(50→1, bias 없음), 교사 부호 과제 | 50 | **추론: 클린 모델과 부호 일치율**, 학습: 라벨 정확도 |
| mnist_linear | 784→128→10 | 101,632 | top-1 |
| mnist_cnn | Conv32→Conv64→FC64→10 | 220,064 | top-1 |
| cifar_deep | 3단 Conv, 32×32 | 307,040 | top-1 |
| resnet18_t10 | ImageNet 사전학습, 백본 동결, fc만 학습, 32×32 입력 그대로 | 11,172,032 | top-1 (clean 39%) |

---

## 2. 점검 결과

### 2-1. 단위 점검 (check_suite.py, 20항목 모두 통과)

| 항목 | 결과 |
|---|---|
| 6개 모듈 컴파일 | 통과 |
| 내장 자가진단 `self_test_bitflip` (1.0의 31번 비트 → −1.0, 30번 → 거대/∞) | 통과 |
| `even_parity_u32` vs 파이썬 `bin().count('1') % 2`, 20,000개 무작위 | 완전 일치 |
| `xor_one_bit`가 살아있는 파라미터를 제자리에서 바꾸는가 (연속/비연속 텐서 모두) | 통과 |
| `leo_saa` λ = r·N_bits·T (T = 1, 7, 30일) | 일치 |
| SAA 1회 통과(`--T saa_pass`) = 하루 선량의 30% | 일치 |
| SAA 구간 비중첩(T=7, 21회) / 이벤트의 SAA 내부 비율 | 비중첩 / 90.1% |
| `events_by_step` t=0→0, 0.3→3, 1.0→9 (10스텝) | 통과 |
| stress 기대 명중 p·N (p=0.01, N=100,000 → 987) | 통과 |
| `write_median([2, ∞, 2])` = 2 | 통과 |
| parity가 뒤집힌 가중치 1개만 탐지·0 처리 | 통과 |
| 초기화 직후 5개 모델(사전학습 ResNet 포함) `max|param|` | 모두 3 미만 → clip이 클린 모델을 건드리지 않음 |

### 2-2. 실제 실행 (16셀, 예외 0건)

| 셀 | 결과 | 확인한 것 |
|---|---|---|
| toy50 infer stress p=0.05 × none/clip/tmr/ensemble/parity | 95.6 / 98.9 / 99.2 / **77.4 ± 22.8** / 96.5 | 5개 방어 경로 모두 동작. 앙상블 이상 현상(2-3 ③) |
| toy50 train stress p=0.05 × 5방어 | 89.5 / 87.3 / 88.7 / 79.1(crash 1/3) / 88.3 | 학습 경로 5개 동작 |
| toy50 infer leo_saa T=7 r=1e-3 | 76.5, k=10 | 다일 노출·높은 r 경로 |
| mnist_linear infer stress 1% none / clip (n_train 8000) | 18.2 / 86.2 (clean 87.4) | MNIST 다운로드·체크포인트 생성·재사용 |
| mnist_linear train stress 1% none | crash 2/2, acc 10.0 | 붕괴 판정 경로 |
| mnist_linear infer leo_saa T=30 none | 56.1 ± 19.3, k≈98 | 장기 임무 경로 |

### 2-3. 고쳐야 할 것 (논문 실험 전)

**① 하루의 정수배가 아닌 T에서는 λ ≠ r·N_bits·T** (`radiation.sample_leo_saa_events`)
SAA 통과 횟수를 `round(3T)`로 잡기 때문에 T가 하루의 정수배가 아니면 선량이 비례하지 않는다. 실측: T=0.5 → 1.30배, T=1.5 → 0.90배, T=2.5 → 1.06배, T=1/120(한 번 통과) → 36배(설계 의도). README의 "λ = r × N_bits × Δt" 문장은 정수 일수에서만 맞다.
→ 논문 E3는 T ∈ {1, 7, 30, 90}처럼 **정수 일수만** 쓰거나, 분수 일수는 `poisson` 모드를 쓴다. README에 한 줄 추가.

**② 앙상블은 "거대하지만 유한한" 로짓에 무력하다** (`defenses.nan_to_num_logits`, `engine.evaluate_ensemble_logits`)
로짓의 NaN/∞만 0으로 바꾸므로, 30번 비트가 뒤집혀 가중치가 약 2^127이 된 복제본은 샘플에 따라 ∞(→0)와 10^38(→그대로)을 섞어 내놓고, 10^38짜리 로짓이 평균을 지배한다. 실측(toy50, p=0.05): 복제본 4개가 100%인데 앙상블 64% (seed 4), 56% (seed 11). 파일럿에서 toy50 stress ensemble(96.5)이 none(99.3)보다 낮았던 이유가 이것이다. 수업 15주차의 `defense_ensemble_safe`도 같은 약점이 있다(가중치의 ∞만 정리).
→ 선택지: (a) 로짓을 [−C, C]로 clip한 뒤 평균, (b) 로짓 중앙값, (c) 복제본별 예측의 다수결. 논문에서는 "평균은 유한한 이상치 하나에도 무너진다"는 **결과**로 쓰고, 개선판을 하나 추가하는 것이 가장 좋다.

**③ GPU로 옮기면 주입이 조용히 사라진다** (`radiation.xor_one_bit`, `apply_stress_inplace`)
`.cpu().numpy()`는 CPU 텐서에서만 메모리를 공유한다. CUDA 텐서면 복사본만 뒤집히고 모델은 그대로다. 지금은 모델을 GPU로 보내는 코드가 없어 문제없지만, 학생이 Colab GPU에서 돌리면 "방사선을 쏴도 멀쩡한" 가짜 결과가 나온다.
→ `xor_one_bit` 첫 줄에 `assert w.device.type == "cpu"` 한 줄 추가.

**④ 넓은 `except Exception`이 코딩 오류를 "붕괴"로 둔갑시킨다** (`engine.run_infer_trial` 357행, `_train_one_model` 510행, `evaluate` 108행)
모양 불일치 같은 진짜 버그가 나도 `crashed=1, acc=10`으로 조용히 기록된다. 파일럿 결과의 crash는 모두 NaN/∞ 경로임을 이번 재실행으로 확인했지만, 코드를 고친 뒤에는 위험하다.
→ `except Exception as exc: traceback.print_exc(); row["note"] = repr(exc)`처럼 남기기.

### 2-4. 개선 권고 (기능에는 문제 없음)

| # | 위치 | 내용 | 권고 |
|---|---|---|---|
| 5 | `simulate.trial_seed` | 시드 = master·10⁵ + hash·20 + trial. 반복이 20회를 넘으면 인접 해시 셀과 시드가 겹칠 수 있음 | 계획대로 30회 돌리려면 `h*20` → `h*1000` |
| 6 | `engine.ckpt_path` | 체크포인트 파일명이 모델 이름만 포함. `--n-train`·epochs를 바꿔도 예전 파일을 그대로 로드 | 파일명에 n_train·epochs 포함하거나 `--force-ckpt` 습관화 |
| 7 | `simulate.run_cell` 208행 | 추론 행의 `epochs` 열이 ckpt_epochs가 아니라 학습 epochs를 기록(메타데이터 오기) | `ckpt_epochs`로 교체 |
| 8 | toy50 | `clean_acc` 열(라벨 정확도 91.8)과 추론 지표(클린 모델 부호 일치, 무피해 시 100)가 다른 척도 | 상대 정확도·생존률 계산 시 toy50은 100을 기준으로 |
| 9 | TMR 학습 | 샘플된 이벤트의 위치·비트를 버리고 복제본마다 k개 새 무작위 플립 → `n_exponent_hits`·`n_sign_hits` 열이 실제 적용과 무관 | TMR 학습 행에서는 두 열을 해석하지 말 것(문서화) |
| 10 | stress 학습 | 수업 코드는 매 epoch마다 p, 스위트는 총 p를 스텝에 분산 → 수업 8 epoch = 스위트의 8배 선량 | 논문 방법 절에 명시 |
| 11 | 앙상블 추론 | 복제본 5개가 각각 전체 선량을 받으므로 시스템 총 명중은 5배, `k` 열은 복제본 평균 | 비용 표에 "메모리 5배 = 명중 5배"로 기록 |
| 12 | `simulate` 그림 | `boxplot(tick_labels=)`는 matplotlib 3.9 이상 전용 | Colab 구버전이면 `labels=`로 |
| 13 | 사소 | `apply_stress_inplace` 미사용, `write_notes`의 `if False` 잔재, `evaluate`는 NaN 출력이 하나라도 있으면 시행 전체를 찍기 수준으로(수업 코드는 샘플별 오답) | 정리 |

### 2-5. 논문 실험을 위해 추가할 옵션 (실험계획서 3절과 동일)

- `--allowed-bits 30` / `--allowed-bits 23-30`: 비트 영역 제한 (E2, E6c). `_emit_events`·`sample_stress_events`의 `rng.integers(0, 32)` 한 곳만 바꾸면 됨
- `--single-hit`: 이벤트 정확히 1개 (E2). `sample_schedule`에 분기 추가
- 요약 CSV에 `median_acc`, `iqr`, `survival_rate`(acc ≥ 0.9·clean) 열 (`make_summary`)
- `--preset paper_v2`

### 2-6. 수정 후 재검증 (2026-09-11, `sim/seu_suite/` 수정본)

| 검증 | 결과 |
|---|---|
| 단위 점검 20항목 재실행 + 자가진단에 추가된 "분수 일수 E[λ] = r·N_bits·T" (T=0.5, 1.5, 2.5 각 400회 평균) | 전부 통과 |
| toy50 `ensemble`(평균) vs `ensemble_median`, stress p=0.05, 12회 | 생존률 **0.58 → 1.00**, 중앙값 96.8 → 100 |
| mnist_linear `--single-hit --allowed-bits 30` 8회 / `--allowed-bits 5` 8회 | 30번: 중앙값 72.7, 생존률 0.50 / 5번: 87.35(무피해), 생존률 1.00 |
| mnist_linear `--allowed-bits exponent` stress 0.1% | 중앙값 19.7, 생존률 0 (지수부만 맞으면 붕괴) |
| mnist_linear leo_saa T=1.5(분수 일수) clip | 정상 실행, k≈5.7 ≈ 1.5 × 3.25 |
| mnist_linear 학습 clip/parity/tmr 1 epoch | 정상, 붕괴 0 |
| 요약·그림 3종·NOTES 생성 (새 CSV 10개 + 8/19 파일럿 670행) | 정상. 파일럿 mnist_cnn LEO none: 평균 86.4 → **중앙값 98.8, 생존률 0.8**로 표시 |
| 체크포인트 파일명 | `mnist_linear_clean_n8000_e2.pt`처럼 예산 포함 |
| 출력 CSV 열 | `allowed_bits`, `single_hit`, `note` 추가, 추론 행 `epochs`=체크포인트 epoch, toy50 `clean_acc`=100 |

---

## 3. 이 PC에서 돌리는 법

```
python -m venv seuvenv
seuvenv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
seuvenv\Scripts\pip install numpy pandas matplotlib
cd sim\seu_suite
..\..\seuvenv\Scripts\python simulate.py --model toy50 --mode infer --rad stress --p 0.05 --defense tmr --trials 20
```

데이터는 `sim/data/`에 내려받는다(MNIST 약 12MB, CIFAR-10 약 170MB, ResNet-18 사전학습 45MB). 파일럿 재현은 `--preset paper_sweep`, CPU로 약 9분.
