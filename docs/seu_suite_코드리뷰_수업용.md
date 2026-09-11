# 🔬 seu_suite 코드 리뷰 (수업용, 단계별)

> **무엇을 읽나:** 우리 반이 여름에 만든 "우주 방사선 자동 실험 프로그램"의 진짜 코드.
> **버전:** 2026-09-11 점검·수정본 (`sim/seu_suite/`). 아래 코드 블록은 그 파일에서 **자동으로 뽑아낸 것**이라 문서와 코드가 항상 같다. 줄 번호는 각 블록 첫 줄에 적혀 있다.
> **읽는 법:** 각 단계는 ① 무엇을 하는 코드인가 → ② 코드 → ③ 줄별 해설 → ④ 수업과의 연결 → ⑤ 생각해 볼 질문 순서다. 9주차 `apply_space_radiation()`을 이해했다면 1~4단계는 복습이다.
> **함께 볼 문서:** `seu_suite/README.md`(옵션 설명), `seu_suite_코드설명_및_점검.md`(점검 보고), `학생용_논문실험_안내서.md`(실험 계획).

---

## 0. 전체 지도

### 파일 6개

| 파일 | 역할 | 한 줄 비유 |
|---|---|---|
| `radiation.py` | 방사선 이벤트(언제, 어느 가중치, 어느 비트)를 뽑고, float32의 비트 하나를 XOR | **주사위와 망치** |
| `defenses.py` | clip / tmr / ensemble / ensemble_median / parity | **방패 5종** |
| `engine.py` | 한 번의 시행(trial)을 실제로 돌림: 추론 시행, 학습 시행, 붕괴 판정, 클린 체크포인트 | **실험대** |
| `models.py` | AI 모델 5개, 데이터 로더, 기본 예산(epoch, 배치 크기 등) | **실험 대상** |
| `presets.py` | 논문용 실험표(어떤 모델 × 모드 × 방사선 × 방어를 몇 번) | **실험 계획표** |
| `simulate.py` | 명령줄 인터페이스, 시드 만들기, CSV 요약, 그림 | **조종석** |

### 한 시행이 흘러가는 길 (추론 모드)

```
simulate.run_single / run_paper_sweep
  └─ run_cell ──── 데이터 로드 ── load_or_train_clean (깨끗한 모델 학습·저장·재사용)
        └─ engine.run_infer_trial(seed …)
              ├─ 1. 새 모델에 깨끗한 가중치 로드
              ├─ 2. collect_weight_params : dim≥2 텐서만 (편향 제외)        ← 2단계
              ├─ 3. weight_bank_meta      : 전역 번호 ↔ (텐서, 위치) 변환표   ← 2단계
              ├─ 4. sample_schedule       : 이벤트 목록 [(t, 가중치번호, 비트)] ← 3~6단계
              ├─ 5. apply_events → xor_one_bit : 비트 하나씩 XOR              ← 1단계
              ├─ 6. 방어 (clip / parity / tmr / ensemble)                    ← 10단계
              └─ 7. evaluate : 정확도, NaN/∞면 붕괴                          ← 8단계
```

학습 모드는 4번에서 뽑은 이벤트를 **옵티마이저 스텝**에 배정하고(7단계), 매 스텝 주입 → 방어 → 5중 유한성 검사 → 갱신을 반복한다(11단계).

---

## 1단계. 숫자를 비트로 보고, 딱 하나만 뒤집기

**무엇을 하나.** 4주차·9주차에서 배운 그대로다. float32 한 칸(32비트)을 **같은 메모리를 가리키는 uint32**로 다시 해석한 뒤, `1 << bit`와 XOR해서 비트 하나만 뒤집는다. 수업 함수와 다른 점은 (a) GPU 텐서를 거부하는 안전장치, (b) 여러 텐서를 하나의 긴 줄(전역 번호)로 다루는 점뿐이다.

```python
# radiation.py  (line 125~)
def _u32_view(w: torch.Tensor) -> np.ndarray:
    """uint32 view that SHARES MEMORY with a contiguous CPU float32 tensor.

    Guard: on a CUDA tensor `.cpu().numpy()` is a copy, so a flip would be
    lost silently and the model would look radiation-proof. Keep models on CPU.
    """
    if w.device.type != "cpu":
        raise RuntimeError(
            "SEU injection needs a CPU tensor (got %s): .cpu().numpy() on a CUDA tensor is a copy, "
            "so the bit-flip would be lost silently." % w.device
        )
    if w.dtype != torch.float32:
        raise TypeError(f"SEU injection expects float32 weights, got {w.dtype}")
    return w.detach().numpy().view(np.uint32).ravel()
```

**줄별 해설**
- `w.device.type != "cpu"` → 예외. CUDA 텐서는 `.numpy()`가 **복사본**을 만들기 때문에, 복사본을 뒤집어 봐야 모델은 그대로다. 예전 코드는 이걸 조용히 넘겨서 "방사선을 맞아도 멀쩡한" 가짜 결과를 낼 수 있었다(점검에서 고침).
- `w.detach().numpy()` → CPU에서는 numpy 배열이 텐서와 **메모리를 공유**한다. `.view(np.uint32)`는 바이트를 그대로 두고 "정수로 읽겠다"는 선언. `.ravel()`은 2차원 행렬을 1차원 줄로 편다(복사 없음).

```python
# radiation.py  (line 168~)
def xor_one_bit(params: Sequence[torch.nn.Parameter], cum: np.ndarray, flat_idx: int, bit: int) -> None:
    """In-place IEEE-754 float32 single-bit XOR on a weight element. True SEU."""
    layer = int(np.searchsorted(cum, flat_idx, side="right") - 1)
    local = int(flat_idx - cum[layer])
    w = params[layer].data
    if not w.is_contiguous():
        w = w.contiguous()
        params[layer].data = w
    u32 = _u32_view(w)
    u32[local] ^= np.uint32(1) << np.uint32(int(bit) & 31)
```

**줄별 해설**
- `cum`은 2단계에서 만드는 누적 개수표 `[0, n0, n0+n1, …]`. `np.searchsorted(cum, flat_idx, side="right") - 1`은 "전역 번호 `flat_idx`가 몇 번째 텐서에 속하는가"를 이진 탐색으로 찾는다. 예: `cum=[0, 100, 130]`, `flat_idx=105` → 1번 텐서, 그 안에서 5번째.
- `is_contiguous()`가 거짓이면 메모리가 흩어져 있어 uint32 뷰를 만들 수 없으므로 연속 복사본으로 바꿔 끼운다.
- 마지막 줄이 9주차의 `bits[i] ^= (1 << pos)`와 완전히 같다. `& 31`은 비트 번호를 0~31로 가두는 안전장치.

**수업과의 연결.** 4주차 `data.view(np.uint32)`, 9주차 `apply_space_radiation()`의 ④단계. 도구가 바뀐 것이 아니라 **같은 도구를 여러 텐서에 걸쳐 쓰는 것**뿐이다.

**자가진단.** 프로그램은 시작할 때마다 아래 검사를 통과해야만 실험을 시작한다.

```python
# radiation.py  (line 475~)
def self_test_bitflip() -> None:
    """Sanity: sign-bit XOR of 1.0 → -1.0; exponent hit is huge/non-finite."""
    t = torch.tensor([1.0], dtype=torch.float32)
    u = t.numpy().view(np.uint32)
    u[0] ^= np.uint32(1) << np.uint32(31)
    assert t.item() == -1.0, t.item()
    t2 = torch.tensor([1.0], dtype=torch.float32)
    u2 = t2.numpy().view(np.uint32)
    u2[0] ^= np.uint32(1) << np.uint32(30)
    assert (not np.isfinite(t2.item())) or abs(t2.item()) > 1e30
    # poisson λ bookkeeping
    rng = np.random.default_rng(0)
    n_bits = 50 * 32
    _ev, lam = sample_poisson_events(1e-6, n_bits, 1.0, rng)
    assert abs(lam - 1e-6 * n_bits) < 1e-18, lam
    assert abs(R_SAA_MULTIPLIER - 36.0) < 1e-9, R_SAA_MULTIPLIER
    # leo_saa: E[λ] must equal r*N_bits*T even for fractional days
    rng = np.random.default_rng(1)
    for T in (0.5, 1.5, 2.5):
        lams = [sample_leo_saa_events(1e-6, 3_252_224, T, rng)[1] for _ in range(400)]
        assert abs(np.mean(lams) / (1e-6 * 3_252_224 * T) - 1.0) < 0.05, (T, np.mean(lams))
    # allowed_bits / single_hit
    s = sample_schedule("stress", n_bits=3200, rng=rng, p=1.0, allowed_bits="30")
    assert s.events and all(e.bit == 30 for e in s.events)
    s = sample_schedule("leo_saa", n_bits=3200, rng=rng, single_hit=True, allowed_bits="exponent")
    assert s.k == 1 and 23 <= s.events[0].bit <= 30
```

**생각해 볼 질문**
1. `1.0`의 31번 비트를 뒤집으면 왜 정확히 `-1.0`이 되나? (부호·지수·가수 중 무엇이 바뀌었나)
2. 30번 비트를 뒤집으면 "거대하거나 ∞"라고 했다. 1.0에서는 ∞, 0.01에서는 거대한 유한값이 되는 이유를 지수 8비트로 설명해 보자.
3. 이 검사에 "가수부 0번 비트를 뒤집으면 값이 거의 안 변한다"는 항목을 추가한다면 어떻게 쓰겠는가?

---

## 2단계. 어느 가중치가 맞을 수 있나 — 가중치 은행

**무엇을 하나.** 모델에는 텐서가 여러 개다(mnist_linear는 가중치 2개 + 편향 2개). 방사선은 "몇 번째 텐서"를 모르고 그냥 메모리 어딘가를 때리므로, 모든 가중치를 **한 줄로 번호를 매긴 은행**처럼 다뤄야 한다.

```python
# radiation.py  (line 98~)
def collect_weight_params(model: nn.Module, flip_bias: bool = False) -> list[torch.nn.Parameter]:
    """Weight tensors (dim >= 2). Biases only if flip_bias. Includes frozen params
    (they still sit in SRAM and can take SEUs)."""
    out: list[torch.nn.Parameter] = []
    for _name, p in model.named_parameters():
        if p.dim() >= 2:
            out.append(p)
        elif flip_bias and p.dim() == 1:
            out.append(p)
    return out
```

```python
# radiation.py  (line 110~)
def n_weight_elements(params: Sequence[torch.nn.Parameter]) -> int:
    return int(sum(int(p.numel()) for p in params))
```

```python
# radiation.py  (line 114~)
def n_bits_of(params: Sequence[torch.nn.Parameter]) -> int:
    return n_weight_elements(params) * BITS_PER_FLOAT32
```

```python
# radiation.py  (line 118~)
def weight_bank_meta(params: Sequence[torch.nn.Parameter]) -> tuple[np.ndarray, int]:
    """Prefix sums of numel, so a global flat index can be mapped to (tensor, local)."""
    sizes = [int(p.numel()) for p in params]
    cum = np.cumsum([0] + sizes).astype(np.int64)
    return cum, int(cum[-1]) if len(cum) else 0
```

**줄별 해설**
- `p.dim() >= 2` → 행렬(Linear)과 커널(Conv)만. 편향(1차원)은 기본 제외. `--flip-bias`를 주면 포함. **이건 우리 연구의 "한계" 항목이다**(편향도 실제 메모리에 있으니까).
- `requires_grad=False`인 동결 파라미터(ResNet 백본)도 포함한다. 학습을 안 해도 메모리에는 있으니 방사선은 맞는다.
- `n_bits = 개수 × 32` → README의 N_bits. λ = r × N_bits × T의 그 N_bits.
- `np.cumsum([0] + sizes)` → `[0, 100352, 101632]` 같은 누적표. 1단계 `xor_one_bit`가 이 표로 전역 번호를 (텐서, 위치)로 바꾼다.

**수업과의 연결.** 13주차 "통제 변수: 모델 구조(가중치 50개)". 여기서는 모델마다 N_bits가 다르므로 **같은 확률 p라도 명중 개수 k = p·N_w가 다르다**. 논문 그림 2에서 x축을 k로 바꾸는 이유.

**생각해 볼 질문**
1. mnist_linear의 가중치 개수 101,632를 784×128 + 128×10으로 직접 계산해 보자. 편향 138개는 어디서 왔나?
2. `cum=[0, 100352, 101632]`일 때 전역 번호 100,400은 어느 텐서의 몇 번째인가?

---

## 3단계. 방사선 이벤트 하나 = (시각, 가중치 번호, 비트 번호)

**무엇을 하나.** 수업에서는 "확률 p로 명중 마스크"를 만들었지만, 논문 프로그램은 **이벤트 목록**을 먼저 만들고 나중에 하나씩 적용한다. 그래야 (a) 시각을 학습 스텝에 배정할 수 있고, (b) 몇 번 맞았는지·어느 비트에 맞았는지 기록할 수 있다.

```python
# radiation.py  (line 69~)
@dataclass
class SeuEvent:
    """One SEU: XOR `bit` of flippable weight element `weight_index` at time `t_days`."""

    t_days: float
    weight_index: int
    bit: int  # 0..31
```

```python
# radiation.py  (line 78~)
@dataclass
class RadiationSchedule:
    """Sampled SEU events plus the λ / rate bookkeeping used to generate them."""

    mode: str
    events: list[SeuEvent] = field(default_factory=list)
    lam: float = 0.0
    T_days: float = 1.0
    r: float = DEFAULT_R
    p: float = DEFAULT_STRESS_P
    n_bits: int = 0
    n_weights: int = 0
    n_saa_passes: int = 0
    saa_intervals: list[tuple[float, float]] = field(default_factory=list)

    @property
    def k(self) -> int:
        return len(self.events)
```

```python
# radiation.py  (line 240~)
def _sample_bits(k: int, rng: np.random.Generator, allowed_bits=None) -> np.ndarray:
    """k bit positions, uniform over 0-31 or over `allowed_bits`."""
    if allowed_bits is None:
        return rng.integers(0, 32, size=k, endpoint=False)
    return rng.choice(np.asarray(list(allowed_bits), dtype=np.int64), size=k)
```

```python
# radiation.py  (line 247~)
def _emit_events(
    times: np.ndarray,
    n_weights: int,
    rng: np.random.Generator,
    allowed_bits=None,
) -> list[SeuEvent]:
    k = int(times.size)
    if k == 0 or n_weights <= 0:
        return []
    widx = rng.integers(0, n_weights, size=k, endpoint=False)
    bits = _sample_bits(k, rng, allowed_bits)
    return [
        SeuEvent(t_days=float(t), weight_index=int(w), bit=int(b))
        for t, w, b in zip(times.tolist(), widx.tolist(), bits.tolist())
    ]
```

```python
# radiation.py  (line 141~)
def parse_allowed_bits(spec) -> list[int] | None:
    """'30' | '23-30' | '31,30' | 'sign' | 'exponent' | 'mantissa' | 'all' | None -> list of bit positions."""
    if spec is None:
        return None
    if isinstance(spec, (list, tuple)):
        bits = [int(b) for b in spec]
    else:
        s = str(spec).strip().lower()
        if s in ("", "all", "none"):
            return None
        named = {"sign": [31], "exponent": list(range(23, 31)), "mantissa": list(range(0, 23))}
        if s in named:
            return named[s]
        bits = []
        for part in s.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-")
                bits.extend(range(int(lo), int(hi) + 1))
            elif part:
                bits.append(int(part))
    bits = sorted(set(bits))
    if not bits or min(bits) < 0 or max(bits) > 31:
        raise ValueError(f"allowed_bits must be within 0..31, got {spec!r}")
    return bits
```

**줄별 해설**
- `SeuEvent` 는 값 3개짜리 작은 상자(dataclass). `t_days`는 0~T 사이의 시각(일 단위).
- `RadiationSchedule.k` 는 이벤트 개수 = CSV의 `k` 열.
- `_sample_bits` : 기본은 0~31 균등. `allowed_bits=[30]`이면 30번만 → **비트 치명도 실험(E2)** 용. `parse_allowed_bits`가 `"30"`, `"23-30"`, `"exponent"` 같은 문자열을 목록으로 바꿔 준다.
- `_emit_events` : 가중치 번호는 `0 ~ n_weights-1` 균등. 같은 가중치가 두 번 맞을 수도 있다(현실도 그렇다).

**생각해 볼 질문**
1. 한 가중치가 같은 비트에 두 번 맞으면 어떻게 되나? (XOR을 두 번 하면?)
2. `parse_allowed_bits("sign")`의 결과는? `"23-30"`은 몇 개의 비트인가?

---

## 4단계. 수업 방식 = 가속 시험 (`stress`)

**무엇을 하나.** 13·15주차 실습 그대로: 가중치 하나하나가 독립적으로 확률 p로 맞는다. 다만 시각을 0~T 사이에 무작위로 붙여 두어 학습 모드에서도 쓸 수 있게 했다.

```python
# radiation.py  (line 391~)
def sample_stress_events(
    n_weights: int,
    p: float,
    rng: np.random.Generator,
    T_days: float = 1.0,
    allowed_bits=None,
) -> list[SeuEvent]:
    """Independent per-weight hit; event times uniform in [0, T] so training can map them to steps."""
    if n_weights <= 0 or p <= 0:
        return []
    mask = rng.random(int(n_weights)) < float(p)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    bits = _sample_bits(int(idx.size), rng, allowed_bits)
    times = rng.uniform(0.0, max(float(T_days), 1e-12), size=idx.size)
    return [
        SeuEvent(t_days=float(t), weight_index=int(w), bit=int(b))
        for t, w, b in zip(times.tolist(), idx.tolist(), bits.tolist())
    ]
```

**줄별 해설**
- `rng.random(n) < p` 가 13주차의 `dice < probability`. `np.flatnonzero`가 `np.where(mask)[0]`.
- 기대 명중 수는 p·N_w. λ(포아송 평균)가 아니라는 점을 `sample_schedule`이 주석으로 남긴다.

**수업과의 연결.** 이 모드의 p를 궤도 시간으로 바꾸는 식이 `T_eq = p / (32 r)`. p=1%가 r=1e-6에서 312일. **"수업의 20% = 17년"** 이 여기서 나온다.

**생각해 볼 질문**
1. `p=1.0`이면 몇 개가 맞나? 이때 `stress`와 `single_hit`는 어떻게 다른가?
2. 수업 11주차 코드는 **매 epoch 끝마다** p로 뒤집었다. 이 프로그램은 총 p·N_w번을 **전체 학습에 한 번** 분산한다. 8 epoch 학습이면 선량이 몇 배 차이 나는가? 논문에 어떻게 적어야 하나?

---

## 5단계. 물리 방식 = 포아송 과정 (`poisson`)

**무엇을 하나.** 진짜 위성에서는 "비트 하나가 하루에 뒤집힐 확률 r"이 주어진다. N_bits개의 비트를 T일 동안 두면 기대 명중 수 λ = r × N_bits × T이고, 실제 개수 K는 **포아송 분포**를 따른다. 4주차의 "주사위를 아주 많이, 아주 작은 확률로" 굴린 극한이 포아송이다.

```python
# radiation.py  (line 52~66)
BITS_PER_FLOAT32 = 32
DEFAULT_R = 1e-6  # SEU / bit / day
DEFAULT_STRESS_P = 0.01

SAA_PASS_MINUTES = 12.0
SAA_PASSES_PER_DAY = 3
MINUTES_PER_DAY = 1440.0
SAA_FRACTION_OF_DAILY = 0.90  # ~90% of expected daily SEUs in SAA passes

SAA_PASS_DAYS = SAA_PASS_MINUTES / MINUTES_PER_DAY  # 12/1440 = 1/120 day
SAA_TIME_FRACTION = SAA_PASSES_PER_DAY * SAA_PASS_DAYS  # 0.025
# r_saa ≈ r * 0.90 / 0.025 = 36 r
R_SAA_MULTIPLIER = SAA_FRACTION_OF_DAILY / SAA_TIME_FRACTION
# remainder of fluence spread over quiet fraction 0.975
R_QUIET_MULTIPLIER = (1.0 - SAA_FRACTION_OF_DAILY) / (1.0 - SAA_TIME_FRACTION)
```

```python
# radiation.py  (line 264~)
def sample_poisson_events(
    r: float,
    n_bits: int,
    T_days: float,
    rng: np.random.Generator,
    allowed_bits=None,
) -> tuple[list[SeuEvent], float]:
    """Homogeneous Poisson process on [0, T]. λ = r * N_bits * T."""
    T_days = float(max(T_days, 0.0))
    n_weights = int(n_bits) // BITS_PER_FLOAT32
    lam = float(r) * float(n_bits) * T_days
    K = int(rng.poisson(lam)) if lam > 0 and n_weights > 0 else 0
    if K == 0:
        return [], lam
    times = rng.uniform(0.0, T_days, size=K)
    return _emit_events(times, n_weights, rng, allowed_bits), lam
```

**줄별 해설**
- `DEFAULT_R = 1e-6` : **가정값**이다. 실제 부품·궤도에 따라 10배 크거나 작다. 논문 E3에서 1e-7, 1e-5도 돌리는 이유.
- `rng.poisson(lam)` : 평균 λ인 포아송 난수. λ=3.25면 대략 0~8 사이가 나온다.
- 시각은 `[0, T]` 균등 → "언제 맞을지 모른다".

**생각해 볼 질문**
1. λ=3.25일 때 "하루 동안 한 번도 안 맞을 확률"은 e^(−3.25) ≈ 0.039다. 이 공식이 어디서 왔는지 4주차 확률로 설명해 보자.
2. toy50(N_bits=1,600)은 λ=0.0016. 이 값이 우리 수업 실험이 "가속 시험"이었다는 증거인 이유는?

---

## 6단계. 진짜 궤도 = 남대서양 이상 구역 (`leo_saa`)

**무엇을 하나.** 저궤도 위성은 하루에 약 3번, 12분씩 **남대서양 이상(SAA)** 구역을 지나며 그때 방사선을 집중적으로 맞는다(하루 선량의 약 90%). 프로그램은 (1) 하루 안에 통과 구간 3개를 겹치지 않게 배치하고, (2) 통과 구간에는 평균의 36배, 나머지 시간에는 0.1배의 비율로 포아송 난수를 뽑는다.

```python
# radiation.py  (line 282~)
def _place_saa_passes(
    T_days: float,
    n_passes: int,
    pass_dur: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Non-overlapping pass start times: one pass per equally spaced slot, jittered."""
    T_days = float(T_days)
    n_passes = int(n_passes)
    pass_dur = float(pass_dur)
    if n_passes <= 0:
        return np.zeros(0, dtype=np.float64)
    if n_passes * pass_dur >= T_days - 1e-15:
        dur = T_days / n_passes
        return np.arange(n_passes, dtype=np.float64) * dur
    slot = T_days / n_passes
    starts = np.empty(n_passes, dtype=np.float64)
    for i in range(n_passes):
        lo = i * slot
        hi = (i + 1) * slot - pass_dur
        starts[i] = float(lo if hi <= lo else rng.uniform(lo, hi))
    return starts
```

```python
# radiation.py  (line 323~)
def _complement_intervals(T_days: float, occupied: list[tuple[float, float]]) -> list[tuple[float, float]]:
    quiet: list[tuple[float, float]] = []
    t = 0.0
    for lo, hi in sorted(occupied):
        if lo > t:
            quiet.append((t, lo))
        t = max(t, hi)
    if t < T_days:
        quiet.append((t, T_days))
    return quiet
```

```python
# radiation.py  (line 306~)
def _sample_times_in_intervals(
    intervals: list[tuple[float, float]],
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n <= 0 or not intervals:
        return np.zeros(0, dtype=np.float64)
    durs = np.array([hi - lo for lo, hi in intervals], dtype=np.float64)
    total = float(durs.sum())
    if total <= 0:
        return np.zeros(0, dtype=np.float64)
    starts = np.array([lo for lo, _hi in intervals], dtype=np.float64)
    choices = rng.choice(len(intervals), size=n, p=durs / total)
    offs = rng.random(n) * durs[choices]
    return starts[choices] + offs
```

```python
# radiation.py  (line 335~)
def sample_leo_saa_events(
    r: float,
    n_bits: int,
    T_days: float,
    rng: np.random.Generator,
    allowed_bits=None,
) -> tuple[list[SeuEvent], float, int, list[tuple[float, float]]]:
    """LEO day with 3 × 12 min SAA passes (scaled to window T_days).

    E[number of passes] = 3 * T exactly, so E[λ] = r * N_bits * T for any T.
    """
    T_days = float(max(T_days, 0.0))
    n_weights = int(n_bits) // BITS_PER_FLOAT32
    lam_avg = float(r) * float(n_bits) * T_days
    if T_days <= 0.0 or n_weights <= 0:
        return [], lam_avg, 0, []

    pass_dur = SAA_PASS_DAYS
    if T_days <= pass_dur + 1e-12:
        # Window no longer than one pass: it sits entirely inside an SAA pass
        # (the `--T saa_pass` scenario). Dose = 36 r * N_bits * T.
        n_passes = 1
        pass_dur = T_days
    else:
        # Integer part + Bernoulli(fractional part) → E[n_passes] = 3 T exactly.
        n_float = SAA_PASSES_PER_DAY * T_days
        n_passes = int(np.floor(n_float)) + int(rng.random() < (n_float - np.floor(n_float)))
        if n_passes * pass_dur > T_days:  # cannot happen for T > pass_dur, kept as a guard
            n_passes = int(np.floor(T_days / pass_dur + 1e-12))

    starts = _place_saa_passes(T_days, n_passes, pass_dur, rng)
    saa_intervals = [(float(s), float(s + pass_dur)) for s in starts.tolist()]
    quiet_intervals = _complement_intervals(T_days, saa_intervals)

    total_saa = float(sum(hi - lo for lo, hi in saa_intervals))
    total_quiet = float(max(0.0, T_days - total_saa))

    r_saa = float(r) * R_SAA_MULTIPLIER
    r_quiet = float(r) * R_QUIET_MULTIPLIER
    lam_saa = r_saa * float(n_bits) * total_saa
    lam_quiet = r_quiet * float(n_bits) * total_quiet
    lam = lam_saa + lam_quiet

    events: list[SeuEvent] = []
    K_saa = int(rng.poisson(lam_saa)) if lam_saa > 0 else 0
    if K_saa:
        times = _sample_times_in_intervals(saa_intervals, K_saa, rng)
        events.extend(_emit_events(times, n_weights, rng, allowed_bits))
    K_q = int(rng.poisson(lam_quiet)) if lam_quiet > 0 else 0
    if K_q:
        times = _sample_times_in_intervals(quiet_intervals, K_q, rng)
        events.extend(_emit_events(times, n_weights, rng, allowed_bits))
    events.sort(key=lambda e: e.t_days)
    return events, lam, n_passes, saa_intervals
```

**줄별 해설**
- `R_SAA_MULTIPLIER = 0.90 / 0.025 = 36` : 하루의 2.5%(36분) 동안 90%를 맞으려면 비율이 36배여야 한다. `R_QUIET_MULTIPLIER = 0.10 / 0.975 ≈ 0.1026`.
- `_place_saa_passes` : 하루를 n등분한 슬롯마다 통과 1개를 무작위 위치에(지터). 슬롯 안에서만 움직이므로 겹치지 않는다.
- **통과 횟수 = 정수부 + 베르누이(소수부)** : T=1.5일이면 4번 또는 5번(평균 4.5). 예전 코드는 `round(3T)`라서 T가 하루의 정수배가 아니면 선량이 비례하지 않았다(T=0.5 → 1.3배). 점검에서 고친 부분.
- `T ≤ 12분`이면 창 전체가 통과 구간 안에 있다고 본다(`--T saa_pass`). 그래서 12분 창의 λ는 하루의 30%.
- `lam = lam_saa + lam_quiet` 가 T=1일 때 정확히 r·N_bits (검증됨).
- 마지막 `events.sort(...)` : 학습 모드에서 시각 순서대로 주입하기 위해.

```python
# radiation.py  (line 413~)
def sample_schedule(
    mode: str,
    *,
    n_bits: int,
    rng: np.random.Generator,
    r: float = DEFAULT_R,
    T_days: float = 1.0,
    p: float = DEFAULT_STRESS_P,
    allowed_bits=None,
    single_hit: bool = False,
) -> RadiationSchedule:
    mode = str(mode).lower().strip()
    n_weights = int(n_bits) // BITS_PER_FLOAT32
    allowed_bits = parse_allowed_bits(allowed_bits)
    sched = RadiationSchedule(
        mode=mode, T_days=float(T_days), r=float(r), p=float(p),
        n_bits=int(n_bits), n_weights=n_weights,
    )
    if single_hit:
        # Exactly one (weight, bit) event, uniform in time — bit-criticality experiments.
        t = rng.uniform(0.0, max(float(T_days), 1e-12), size=1)
        sched.events = _emit_events(t, n_weights, rng, allowed_bits)
        sched.lam = 1.0
        return sched
    if mode == "stress":
        sched.events = sample_stress_events(n_weights, p, rng, T_days=T_days, allowed_bits=allowed_bits)
        sched.lam = float(p) * n_weights  # expected hits, not a Poisson λ
        return sched
    if mode == "poisson":
        events, lam = sample_poisson_events(r, n_bits, T_days, rng, allowed_bits)
        sched.events = events
        sched.lam = lam
        return sched
    if mode in ("leo_saa", "leo", "saa"):
        events, lam, n_passes, intervals = sample_leo_saa_events(r, n_bits, T_days, rng, allowed_bits)
        sched.events = events
        sched.lam = lam
        sched.n_saa_passes = n_passes
        sched.saa_intervals = intervals
        return sched
    raise ValueError(f"unknown radiation mode: {mode!r} (use poisson|leo_saa|stress)")
```

**줄별 해설 (`sample_schedule` = 세 모드의 교통정리)**
- `single_hit=True` 면 모드에 상관없이 **정확히 이벤트 1개**. E2(비트 하나만 뒤집기) 실험용.
- `stress`는 `lam`에 기대 명중 수(p·N_w)를 넣는다 — 포아송 λ가 아니라는 주석에 주의.

**생각해 볼 질문**
1. 하루 3번 × 12분 = 36분에 90%가 몰린다면, 위성 AI는 하루 중 언제 가장 위험한가? 이걸 이용한 방어 아이디어가 있을까? (힌트: 통과 직전에 백업, 직후에 복원)
2. T=0.5일이면 통과 횟수가 1번 또는 2번이다. 왜 "1.5번"으로 고정하지 않고 확률적으로 뽑는가?

---

## 7단계. 시각을 학습 스텝으로 바꾸기

**무엇을 하나.** 학습 모드에서는 "0.3일째에 맞았다"를 "전체 1,000스텝 중 300번째 스텝에 맞았다"로 바꿔야 한다. 비례식 하나면 된다.

```python
# radiation.py  (line 456~)
def events_by_step(events: Sequence[SeuEvent], n_steps: int, T_days: float) -> dict[int, list[SeuEvent]]:
    """Map event times in [0, T] onto optimizer steps: t=0 → step 0, t=T → last step.

    An event at 30% of the window is injected at step floor(0.3 * n_steps).
    """
    from collections import defaultdict

    n_steps = max(int(n_steps), 1)
    T_days = float(T_days) if T_days > 0 else 1.0
    bucket: dict[int, list[SeuEvent]] = defaultdict(list)
    for ev in events:
        frac = min(max(ev.t_days / T_days, 0.0), 1.0 - 1e-15)
        step = int(frac * n_steps)
        if step >= n_steps:
            step = n_steps - 1
        bucket[step].append(ev)
    return dict(bucket)
```

**줄별 해설**
- `frac = t / T` 를 0~1로 가둔 뒤 `int(frac × n_steps)`. t=T(끝)는 마지막 스텝으로.
- 결과는 `{스텝번호: [이벤트, …]}` 사전. 학습 루프가 매 스텝 `by_step.get(global_step, [])`로 꺼내 쓴다.
- **주의(가정):** "학습 전체 = 궤도 1일"은 `--train-hours 24`라는 **가정**이다. 노트북에서 2초 걸린 학습이 실제로 2초 동안 우주에 있었다는 뜻이 아니다. 논문에 반드시 적는다.

**생각해 볼 질문**
1. 총 500스텝, T=1일 때 t=0.999일의 이벤트는 몇 번째 스텝에 들어가나? t=1.0은?

---

## 8단계. 채점과 "죽었다"의 정의

**무엇을 하나.** 망가진 모델의 정확도를 잰다. 가중치나 출력에 NaN/∞가 하나라도 있으면 `None`을 돌려주고, 호출한 쪽이 이를 **붕괴(crash)** 로 기록하고 정확도는 찍기 수준(10클래스면 10%, toy50은 50%)으로 적는다.

```python
# defenses.py  (line 101~)
def params_finite(model: nn.Module) -> bool:
    for p in model.parameters():
        if not torch.isfinite(p.data).all():
            return False
    return True
```

```python
# engine.py  (line 86~)
def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    model_name: str,
    clean_model: nn.Module | None = None,
) -> float | None:
    """Accuracy in percent, or None if non-finite.

    toy50 infer: sign-agreement vs clean predictions (course metric).
    toy50 train: sign match vs binary labels.
    others: top-1 vs labels.
    """
    model.eval()
    if not params_finite(model):
        return None
    bs = _eval_batch_size(model_name)
    n = int(x.shape[0])
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, n, bs):
            xb = x[start : start + bs]
            yb = y[start : start + bs]
            try:
                logits = model(xb)
            except Exception:
                traceback.print_exc()  # a real bug must not hide behind "crash"
                return None
            if not torch.isfinite(logits).all():
                return None
            if model_name == "toy50" and clean_model is not None:
                clean_pred = clean_model(xb)
                if not torch.isfinite(clean_pred).all():
                    return None
                # sign agreement of scalar scores
                a = logits.reshape(-1)
                b = clean_pred.reshape(-1)
                finite = torch.isfinite(a) & torch.isfinite(b)
                match = (torch.sign(a) == torch.sign(b)) & finite
                correct += int(match.sum().item())
                total += int(a.numel())
            elif model_name == "toy50":
                pred = (logits.reshape(-1) >= 0).long()
                correct += int((pred == yb.long()).sum().item())
                total += int(yb.numel())
            else:
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())
    if total == 0:
        return None
    acc = 100.0 * correct / total
    if not np.isfinite(acc):
        return None
    return float(acc)
```

```python
# models.py  (line 184~)
def chance_acc_of(name: str) -> float:
    """Percent. 1/n_classes * 100, used when a run crashes with no finite acc."""
    return 100.0 / n_classes_of(name)
```

**줄별 해설**
- `params_finite(model)` 먼저 → 가중치에 ∞가 있으면 계산도 안 하고 `None`.
- `torch.isfinite(logits).all()` → 출력에 NaN/∞가 하나라도 있으면 `None`. 수업 12주차 `evaluate`는 NaN 출력을 **샘플별로** 오답 처리했지만, 여기서는 **시행 전체**를 붕괴로 본다. Linear 층에서는 ∞ 가중치 하나가 모든 샘플의 출력을 오염시키므로 실제로는 같은 결과다.
- toy50 추론은 라벨이 아니라 **깨끗한 모델과 부호가 같은지**(sign agreement)로 채점한다 — 13주차 `measure_accuracy`와 같은 방식. 그래서 toy50의 clean 기준값은 100이다.
- `except Exception: traceback.print_exc()` : 예전엔 조용히 `None`을 돌려줘서 코딩 실수가 "붕괴"로 둔갑할 수 있었다(점검에서 고침).

**생각해 볼 질문**
1. "붕괴"를 이렇게 정의하면 정확도 분포가 왜 **이봉(bimodal)** 이 되나? 평균 대신 무엇으로 보고해야 하나?
2. 붕괴한 시행의 정확도를 0이 아니라 "찍기 수준"으로 적는 이유는?

---

## 9단계. 추론 시행 한 번 — 12주차의 논문 버전

**무엇을 하나.** 깨끗한 가중치를 불러와 → 이벤트 뽑고 → 주입하고 → 방어하고 → 채점. TMR과 앙상블은 "오염된 복제본 여러 개"가 필요하므로 같은 일을 3번·5번 반복한다.

```python
# engine.py  (line 277~)
def run_infer_trial(
    *,
    model_factory,
    clean_state: dict,
    data,
    model_name: str,
    rad: str,
    defense: str,
    seed: int,
    r: float,
    T_days: float,
    p: float,
    flip_bias: bool,
    allowed_bits=None,
    single_hit: bool = False,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))
    note = ""

    clean_model = model_factory()
    clean_model.load_state_dict(clean_state)
    clean_model.eval()

    def _one_irradiated(rng_i: np.random.Generator) -> tuple[nn.Module, object]:
        m = model_factory()
        m.load_state_dict(copy.deepcopy(clean_state))
        params = collect_weight_params(m, flip_bias=flip_bias)
        cum, n_w = weight_bank_meta(params)
        n_bits = n_w * 32
        sched = _fresh_schedule(rad, n_bits, rng_i, r, T_days, p, allowed_bits, single_hit)
        parity_table = parity_of_params(params) if defense == "parity" else None
        apply_events(params, cum, sched.events)
        if defense == "clip":
            clip_all_params(m)
        elif defense == "parity" and parity_table is not None:
            zero_failed_parity(params, parity_table)
        elif defense == "none":
            pass
        m.eval()
        return m, sched

    n_bits = n_bits_of(collect_weight_params(clean_model, flip_bias=flip_bias))
    crashed = 0
    k_mean = 0.0
    n_exp = n_sign = 0
    acc: float | None = None

    try:
        if defense == "tmr":
            snaps = []
            ks = []
            for i in range(TMR_REPLICAS):
                m, sched = _one_irradiated(np.random.default_rng(int(seed) * 1009 + i + 1))
                snaps.append(snapshot_params(collect_weight_params(m, flip_bias=flip_bias)))
                ks.append(sched.k)
                e, s = _count_bit_kinds(sched.events)
                n_exp += e
                n_sign += s
            voted = model_factory()
            voted.load_state_dict(copy.deepcopy(clean_state))
            write_median(collect_weight_params(voted, flip_bias=flip_bias), snaps)
            if not params_finite(voted):
                clip_all_params(voted)  # last-ditch so eval can run; still a crash if originally nonfinite
            k_mean = float(np.mean(ks)) if ks else 0.0
            n_exp = int(round(n_exp / max(len(ks), 1)))
            n_sign = int(round(n_sign / max(len(ks), 1)))
            acc = evaluate(voted, data.x_test, data.y_test, model_name, clean_model=clean_model)
        elif defense in ENSEMBLE_DEFENSES:
            replicas = []
            ks = []
            for i in range(ENSEMBLE_INFER_REPLICAS):
                m, sched = _one_irradiated(np.random.default_rng(int(seed) * 1009 + i + 1))
                replicas.append(m)
                ks.append(sched.k)
                e, s = _count_bit_kinds(sched.events)
                n_exp += e
                n_sign += s
            k_mean = float(np.mean(ks)) if ks else 0.0
            n_exp = int(round(n_exp / max(len(ks), 1)))
            n_sign = int(round(n_sign / max(len(ks), 1)))
            acc = evaluate_ensemble_logits(
                replicas, data.x_test, data.y_test, model_name, clean_model=clean_model,
                how=ensemble_reduce_of(defense),
            )
        else:
            m, sched = _one_irradiated(rng)
            k_mean = float(sched.k)
            n_exp, n_sign = _count_bit_kinds(sched.events)
            acc = evaluate(m, data.x_test, data.y_test, model_name, clean_model=clean_model)
    except Exception as exc:
        traceback.print_exc()
        note = f"exception: {exc!r}"
        crashed = 1
        acc = None

    if acc is None:
        crashed = 1
        acc = chance_acc_of(model_name)

    return {
        "acc": float(acc),
        "crashed": int(crashed),
        "k": float(k_mean),
        "n_bits": int(n_bits),
        "n_exponent_hits": int(n_exp),
        "n_sign_hits": int(n_sign),
        "seconds": float(time.perf_counter() - t0),
        "last_finite_acc": float(acc),
        "note": note,
    }
```

```python
# engine.py  (line 391~)
def clean_reference_acc(model_factory, clean_state: dict, data, model_name: str) -> float:
    """Clean accuracy measured with the SAME metric the irradiated trials use.

    toy50 infer scores sign-agreement with the clean model, so its reference is 100.
    Other models: top-1 of the clean checkpoint on this test set.
    """
    m = model_factory()
    m.load_state_dict(copy.deepcopy(clean_state))
    m.eval()
    ref = evaluate(m, data.x_test, data.y_test, model_name, clean_model=m if model_name == "toy50" else None)
    return float(ref) if ref is not None else float("nan")
```

**줄별 해설**
- `_one_irradiated(rng_i)` : 12주차 `inference_bitflip_test` 한 회차와 같다. 클린 가중치 복사 → 이벤트 → XOR → clip/parity → 모델 반환.
- **TMR** : 복제본 3개를 각각 독립적으로 오염시킨 뒤 `write_median`으로 **원소별 중앙값** 을 새 모델에 쓴다. 15주차 `defense_tmr`과 동일.
- **앙상블** : 복제본 5개의 출력(로짓)을 합친다. `ensemble`은 평균, `ensemble_median`은 중앙값. (10단계에서 왜 둘이 필요한지 본다.)
- 복제본마다 시드를 `seed*1009 + i + 1`로 달리 주어 서로 다른 위치에 맞게 한다.
- 시행 결과 사전: `acc`, `crashed`, `k`(명중 수), `n_exponent_hits`, `n_sign_hits`, `note`(예외 메시지). CSV 한 줄이 된다.
- `clean_reference_acc` : "깨끗할 때 몇 점인가"를 **시행과 같은 자로** 잰다. toy50은 자기 자신과의 부호 일치이므로 100. 생존률(acc ≥ 0.9 × 기준)의 기준값.

**생각해 볼 질문**
1. 앙상블 복제본 5개가 **각각** 전체 선량을 받는다. 시스템 전체로는 몇 배의 방사선을 맞는 셈인가? 이것이 공정한 비교인가? (메모리 5배 = 비트 5배)
2. TMR 복제본 3개 중 2개가 **같은 가중치**에 맞을 확률은 대략 얼마인가? (N_w=101,632, 각 3번 명중)

---

## 10단계. 방패 다섯 개

**무엇을 하나.** 15주차에서 만든 방어 4종 + 점검에서 추가한 `ensemble_median`. 핵심 수학은 각각 **범위 제한, 중앙값, 덧셈 mod 2, 평균, 중앙값**이다.

### 10-1. Clipping — 큰 값 자르기

```python
# defenses.py  (line 64~)
def clip_all_params(model: nn.Module, lo: float = CLIP_LO, hi: float = CLIP_HI) -> None:
    """Week-11 harden: nan_to_num + clip weights AND biases to [lo, hi]."""
    with torch.no_grad():
        for p in model.parameters():
            p.data.nan_to_num_(nan=0.0, posinf=hi, neginf=lo)
            p.data.clamp_(lo, hi)
```

- 11주차 `harden_weights`와 같다: NaN→0, +∞→3, −∞→−3, 그리고 [−3, 3]로 자름.
- **모든** 파라미터(편향, BatchNorm 포함)를 자른다. 5개 모델 모두 깨끗한 상태에서 |값|<3이므로 클린 모델은 안 바뀐다(점검에서 확인).
- 한계: 명중이 수만 개면 ±3으로 잘린 가중치가 수만 개 → 이미 다른 모델. 파일럿의 ResNet stress 1% = 10점.

### 10-2. TMR — 중앙값은 ∞에 면역

```python
# defenses.py  (line 79~)
def snapshot_params(params: Sequence[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [p.data.detach().clone() for p in params]
```

```python
# defenses.py  (line 83~)
def restore_params(params: Sequence[torch.nn.Parameter], snap: Sequence[torch.Tensor]) -> None:
    with torch.no_grad():
        for p, s in zip(params, snap):
            p.data.copy_(s)
```

```python
# defenses.py  (line 89~)
def write_median(params: Sequence[torch.nn.Parameter], snaps: Sequence[Sequence[torch.Tensor]]) -> None:
    """Per-element median of several snapshots → live params.

    Non-finite replicas are nan_to_num'd before the median so a single Inf
    replica cannot poison the vote (median of finite values is defined).
    """
    with torch.no_grad():
        for i, p in enumerate(params):
            stacked = torch.stack([s[i].nan_to_num(nan=0.0, posinf=0.0, neginf=0.0) for s in snaps], dim=0)
            p.data.copy_(stacked.median(dim=0).values)
```

- `nan_to_num` 후 `median(dim=0)`: 복제본 3개 중 하나가 ∞여도 `median([w, w, 0]) = w`. 15주차 실습 2-3의 표 그대로.
- 학습 모드에서는 주입 전 스냅샷에서 복제본 3개를 만들기 위해 `snapshot_params`/`restore_params`를 쓴다(11단계).

### 10-3. Parity — 1의 개수가 짝수인가

```python
# defenses.py  (line 108~)
def even_parity_u32(u32: np.ndarray) -> np.ndarray:
    """Even-parity bit of each uint32: popcount mod 2. 0 = even number of 1s."""
    x = np.asarray(u32, dtype=np.uint32).ravel()
    # SWAR popcount
    y = x.astype(np.uint32)
    y = y - ((y >> np.uint32(1)) & np.uint32(0x55555555))
    y = (y & np.uint32(0x33333333)) + ((y >> np.uint32(2)) & np.uint32(0x33333333))
    y = (y + (y >> np.uint32(4))) & np.uint32(0x0F0F0F0F)
    y = y + (y >> np.uint32(8))
    y = y + (y >> np.uint32(16))
    return (y & np.uint32(0x3F) & np.uint32(1)).astype(np.uint8)
```

```python
# defenses.py  (line 121~)
def parity_of_params(params: Sequence[torch.nn.Parameter]) -> list[np.ndarray]:
    tables: list[np.ndarray] = []
    for p in params:
        u32 = p.data.detach().cpu().contiguous().numpy().view(np.uint32).ravel()
        tables.append(even_parity_u32(u32))
    return tables
```

```python
# defenses.py  (line 138~)
def zero_failed_parity(params: Sequence[torch.nn.Parameter], stored: Sequence[np.ndarray]) -> int:
    """Inference repair: zero any weight whose even parity no longer matches."""
    n = 0
    with torch.no_grad():
        for p, s in zip(params, stored):
            u32 = p.data.detach().cpu().contiguous().numpy().view(np.uint32).ravel()
            fail = even_parity_u32(u32) != np.asarray(s).ravel()
            if not fail.any():
                continue
            flat = p.data.view(-1)
            idx = torch.from_numpy(np.flatnonzero(fail).astype(np.int64))
            flat[idx] = 0.0
            n += int(fail.sum())
    return n
```

```python
# defenses.py  (line 154~)
def revert_failed_parity(
    params: Sequence[torch.nn.Parameter],
    stored: Sequence[np.ndarray],
    pre_snap: Sequence[torch.Tensor],
) -> int:
    """Training repair: revert parity-failing weights to the pre-flip snapshot."""
    n = 0
    with torch.no_grad():
        for p, s, pre in zip(params, stored, pre_snap):
            u32 = p.data.detach().cpu().contiguous().numpy().view(np.uint32).ravel()
            fail = even_parity_u32(u32) != np.asarray(s).ravel()
            if not fail.any():
                continue
            flat = p.data.view(-1)
            pre_flat = pre.reshape(-1)
            idx = torch.from_numpy(np.flatnonzero(fail).astype(np.int64))
            flat[idx] = pre_flat[idx]
            n += int(fail.sum())
    return n
```

- `even_parity_u32`는 "32비트 중 1이 몇 개인가(popcount)"를 **비트 병렬 트릭(SWAR)** 으로 센다. 2비트씩 → 4비트씩 → 8비트씩 합쳐 올라간다. 마지막 `& 1`이 홀짝. (점검: 무작위 20,000개에서 `bin(x).count('1') % 2`와 완전 일치.)
- 비트 하나가 뒤집히면 1의 개수 홀짝이 반드시 바뀐다 → **단일 비트플립은 100% 검출**. 같은 가중치가 두 번 맞으면 놓친다.
- 추론: 검출된 가중치를 **0으로**(복구는 못 하니 무력화). 학습: 주입 전 스냅샷 값으로 **복원**(11단계에서 스텝마다 패리티 표를 갱신).
- 비용: 가중치당 1비트 = 메모리 1/32 ≈ 3%.

### 10-4. Ensemble — 평균 vs 중앙값 (점검에서 발견한 것)

```python
# defenses.py  (line 175~)
def nan_to_num_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
```

```python
# defenses.py  (line 51~)
def combine_logits(stack: torch.Tensor, how: str = "mean") -> torch.Tensor:
    """stack: (n_replicas, B, C) finite logits → (B, C).

    'mean'   arithmetic mean — one huge-but-finite replica dominates.
    'median' per-element median — robust to a minority of wild replicas.
    """
    if how == "median":
        return stack.median(dim=0).values
    if how == "mean":
        return stack.mean(dim=0)
    raise ValueError(f"unknown ensemble reduce {how!r}")
```

```python
# engine.py  (line 144~)
def evaluate_ensemble_logits(
    replicas: list[nn.Module],
    x: torch.Tensor,
    y: torch.Tensor,
    model_name: str,
    clean_model: nn.Module | None = None,
    how: str = "mean",
) -> float | None:
    """Combine logits across independently corrupted replicas ('mean' or 'median')."""
    for m in replicas:
        m.eval()
        if not params_finite(m):
            # still try: nan_to_num on logits later
            pass
    bs = _eval_batch_size(model_name)
    n = int(x.shape[0])
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, n, bs):
            xb = x[start : start + bs]
            yb = y[start : start + bs]
            outs = []
            for m in replicas:
                try:
                    logits = m(xb)
                except Exception:
                    traceback.print_exc()
                    continue
                outs.append(nan_to_num_logits(logits))
            if not outs:
                return None
            logits = combine_logits(torch.stack(outs, dim=0), how)
            if model_name == "toy50" and clean_model is not None:
                clean_pred = clean_model(xb).reshape(-1)
                a = logits.reshape(-1)
                finite = torch.isfinite(a) & torch.isfinite(clean_pred)
                match = (torch.sign(a) == torch.sign(clean_pred)) & finite
                correct += int(match.sum().item())
                total += int(a.numel())
            elif model_name == "toy50":
                pred = (logits.reshape(-1) >= 0).long()
                correct += int((pred == yb.long()).sum().item())
                total += int(yb.numel())
            else:
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())
    if total == 0:
        return None
    return float(100.0 * correct / total)
```

- 복제본마다 출력(로짓)을 구하고 NaN/∞를 0으로 바꾼 뒤 `combine_logits`로 합친다.
- **문제:** 30번 비트에 맞은 가중치는 ∞가 아니라 약 2^127 ≈ 10^38이 될 수 있다. 그러면 그 복제본의 로짓은 샘플에 따라 ∞(→0)와 10^38(→그대로!)이 섞이고, 10^38이 평균을 지배한다. 점검 실측: 복제본 4개가 100점인데 **평균 앙상블 64점**. 15주차 `defense_ensemble_safe`도 같은 약점이 있다.
- **해결:** `ensemble_median` = 로짓의 원소별 중앙값. 실측: 같은 조건에서 평균 앙상블 생존률 58% → 중앙값 앙상블 100%.
- 15주차의 결론 "평균은 이상치 하나에 무너지고 중앙값은 버틴다"가 **출력 단계에서도** 그대로 성립한다.

### 10-5. 학습 중 방어의 교통정리

```python
# engine.py  (line 226~)
def apply_defense_after_injection(
    *,
    defense: str,
    model: nn.Module,
    params,
    cum,
    n_weights: int,
    pending: list[SeuEvent],
    rng: np.random.Generator,
    parity_table,
    pre_snap,
    mode: str,
    p: float,
    allowed_bits=None,
) -> int:
    """Apply `defense` to live weights AFTER `pending` events have been XOR'd.

    Returns extra flip count introduced by TMR's independent replicas
    (already applied; the live W is the median, so they are not "on" W).
    """
    extra = 0
    if defense == "none":
        return extra
    if defense == "clip":
        clip_all_params(model)
        return extra
    if defense == "parity":
        if parity_table is not None and pre_snap is not None:
            revert_failed_parity(params, parity_table, pre_snap)
        return extra
    if defense == "tmr":
        # 3 independent corruptions of the *pre-injection* snapshot, median → W.
        # Same event count K (including stress hits assigned to this step),
        # independent locations/bits — not a second full p-pass.
        k = len(pending)
        snaps = []
        for _ in range(TMR_REPLICAS):
            restore_params(params, pre_snap)
            extra += len(_independent_k_flips(params, cum, n_weights, k, rng, allowed_bits))
            snaps.append(snapshot_params(params))
        write_median(params, snaps)
        return extra
    if defense in ENSEMBLE_DEFENSES:
        # Training ensemble is handled at the replica-loop level, not here.
        return extra
    raise ValueError(f"unknown defense {defense!r}")
```

- 학습 루프가 이벤트를 주입한 직후 호출한다. clip은 전체 자르기, parity는 스냅샷 복원, tmr은 스냅샷에서 복제본 3개를 만들어 중앙값.
- tmr 학습은 뽑아 둔 이벤트의 위치·비트를 쓰지 않고 복제본마다 k개의 **새 무작위** 플립을 준다. 그래서 tmr 학습 행의 `n_exponent_hits` 열은 실제 적용과 무관(README "알아둘 것").

**생각해 볼 질문**
1. 메모리 3배(TMR)와 5배(앙상블) 중 어느 쪽이 위성에 실을 만한가? 생존률 결과와 함께 판단해 보자.
2. 패리티가 **두 번** 맞은 가중치를 놓치는 확률을 계산해 보자. (N_w=101,632, 하루 3번 명중)
3. `combine_logits`에 "가장 큰 값과 가장 작은 값을 버리고 평균"(트림 평균)을 추가한다면 어떻게 짜겠는가?

---

## 11단계. 학습 중 공격 — 11주차의 논문 버전

**무엇을 하나.** 11주차 B군·C군 실험을 일반화한 것. 이벤트를 스텝에 배정해 두고, 매 스텝 "주입 → 방어 → 5중 유한성 검사 → 역전파 → 갱신"을 반복한다. 어디서든 NaN/∞가 나오면 즉시 멈추고 `crashed=1`.

```python
# engine.py  (line 407~)
def _n_batches(n_train: int, batch_size: int) -> int:
    n = n_train // batch_size
    rem = n_train % batch_size
    if rem >= 2:
        n += 1
    return max(n, 1)
```

```python
# engine.py  (line 422~)
def _train_one_model(
    *,
    model: nn.Module,
    data,
    model_name: str,
    rad: str,
    defense: str,
    seed: int,
    rng: np.random.Generator,
    r: float,
    T_days: float,
    p: float,
    flip_bias: bool,
    epochs: int,
    batch_size: int,
    lr: float,
    momentum: float,
    replica_tag: int = 0,
    allowed_bits=None,
    single_hit: bool = False,
) -> dict[str, Any]:
    """Single-network training with in-graph injection. Used for none/clip/tmr/parity
    and as one replica of a training ensemble."""
    note = ""
    opt = _make_opt(model, lr, momentum)
    # toy50: BCE-with-logits on a scalar score; others: CE
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    params = collect_weight_params(model, flip_bias=flip_bias)
    cum, n_weights = weight_bank_meta(params)
    n_bits = n_weights * 32
    sched = _fresh_schedule(rad, n_bits, rng, r, T_days, p, allowed_bits, single_hit)

    n_train = int(data.x_train.shape[0])
    steps_per_epoch = _n_batches(n_train, batch_size)
    total_steps = max(epochs * steps_per_epoch, 1)
    by_step = events_by_step(sched.events, total_steps, sched.T_days if sched.T_days > 0 else 1.0)

    parity_table = parity_of_params(params) if defense == "parity" else None
    n_exp, n_sign = _count_bit_kinds(sched.events)

    crashed = 0
    n_flips_applied = 0
    last_finite_acc: float | None = None
    global_step = 0
    x_train, y_train = data.x_train, data.y_train

    try:
        for _epoch in range(int(epochs)):
            model.train()
            perm = torch.randperm(n_train)
            for start in range(0, n_train, batch_size):
                idx = perm[start : start + batch_size]
                if idx.numel() < 2:
                    continue
                pending = by_step.get(global_step, [])
                pre_snap = snapshot_params(params) if pending else None
                if pending and defense != "tmr":
                    apply_events(params, cum, pending)
                    n_flips_applied += len(pending)
                if pending:
                    extra = apply_defense_after_injection(
                        defense=defense if defense not in ENSEMBLE_DEFENSES else "none",
                        model=model,
                        params=params,
                        cum=cum,
                        n_weights=n_weights,
                        pending=pending,
                        rng=rng,
                        parity_table=parity_table,
                        pre_snap=pre_snap,
                        mode=rad,
                        p=p,
                        allowed_bits=allowed_bits,
                    )
                    if defense == "tmr" and pending:
                        n_flips_applied += len(pending)  # nominal K; live W is the median
                    n_flips_applied += extra

                if not params_finite(model):
                    crashed = 1
                    break

                opt.zero_grad(set_to_none=True)
                logits = model(x_train[idx])
                if not torch.isfinite(logits).all():
                    crashed = 1
                    break
                if model_name == "toy50":
                    loss = bce(logits.reshape(-1), y_train[idx].float())
                else:
                    loss = ce(logits, y_train[idx])
                if not torch.isfinite(loss):
                    crashed = 1
                    break
                loss.backward()
                grads_ok = True
                for p_ in model.parameters():
                    if p_.grad is not None and not torch.isfinite(p_.grad).all():
                        grads_ok = False
                        break
                if not grads_ok:
                    crashed = 1
                    break
                opt.step()
                if not params_finite(model):
                    crashed = 1
                    break
                if defense == "parity":
                    parity_table = parity_of_params(params)
                global_step += 1
            if crashed:
                break
            acc_e = evaluate(model, data.x_test, data.y_test, model_name, clean_model=None)
            if acc_e is None:
                crashed = 1
                break
            last_finite_acc = acc_e
    except Exception as exc:
        traceback.print_exc()
        note = f"exception: {exc!r}"
        crashed = 1

    return {
        "model": model,
        "crashed": int(crashed),
        "k": float(n_flips_applied if defense != "tmr" else sched.k),
        "n_bits": int(n_bits),
        "n_exponent_hits": int(n_exp),
        "n_sign_hits": int(n_sign),
        "last_finite_acc": last_finite_acc,
        "sched_k": int(sched.k),
        "note": note,
    }
```

**줄별 해설 (루프 안쪽 순서가 핵심)**
1. `pending = by_step.get(global_step, [])` : 이 스텝에 배정된 이벤트.
2. `pre_snap` : 이벤트가 있으면 주입 전 가중치를 저장(parity 복원·tmr 복제본용).
3. `apply_events` → `apply_defense_after_injection` : 주입 후 방어. (tmr은 주입 대신 복제본 방식.)
4. **5중 검사**: 가중치 → 로짓 → 손실 → 그래디언트 → 갱신 후 가중치. 하나라도 NaN/∞면 `crashed=1; break`. 11주차 C군에서 본 "NaN 즉사"를 코드로 잡아낸 것.
5. `parity` 는 정상 갱신 뒤 패리티 표를 새로 계산(가중치가 합법적으로 바뀌었으니까).
6. epoch마다 테스트 정확도를 재서 `last_finite_acc`에 남긴다 → 붕괴 시 "마지막으로 살아 있던 점수".
- `except Exception as exc: traceback.print_exc(); note = ...` : 진짜 오류는 화면과 CSV `note` 열에 남는다.

```python
# engine.py  (line 559~)
def run_train_trial(
    *,
    model_factory,
    data,
    model_name: str,
    rad: str,
    defense: str,
    seed: int,
    r: float,
    T_days: float,
    p: float,
    flip_bias: bool,
    epochs: int,
    batch_size: int,
    lr: float,
    momentum: float,
    allowed_bits=None,
    single_hit: bool = False,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    chance = chance_acc_of(model_name)

    if defense in ENSEMBLE_DEFENSES and model_name not in ENSEMBLE_TRAIN_MODELS:
        # Documented: CNN/ResNet ensemble is inference-only. Fall back to none
        # during train so the cell still produces a number, and tag it.
        defense_eff = "none"
        ensemble_skipped = 1
    else:
        defense_eff = defense
        ensemble_skipped = 0

    if defense_eff in ENSEMBLE_DEFENSES:
        replicas = []
        infos = []
        for i in range(ENSEMBLE_TRAIN_REPLICAS):
            m = model_factory()
            info = _train_one_model(
                model=m,
                data=data,
                model_name=model_name,
                rad=rad,
                defense="none",  # each replica is independently irradiated
                seed=int(seed),
                rng=np.random.default_rng(int(seed) * 1009 + i + 7),
                r=r,
                T_days=T_days,
                p=p,
                flip_bias=flip_bias,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                momentum=momentum,
                replica_tag=i,
                allowed_bits=allowed_bits,
                single_hit=single_hit,
            )
            replicas.append(info["model"])
            infos.append(info)
        crashed = int(any(i["crashed"] for i in infos))  # "crash" = any replica died
        acc = evaluate_ensemble_logits(replicas, data.x_test, data.y_test, model_name, clean_model=None,
                                       how=ensemble_reduce_of(defense_eff))
        last = None
        for i in infos:
            if i["last_finite_acc"] is not None:
                last = i["last_finite_acc"]
        if acc is None:
            crashed = 1
            acc = last if last is not None else chance
        k_mean = float(np.mean([i["sched_k"] for i in infos]))
        n_bits = int(infos[0]["n_bits"])
        n_exp = int(round(np.mean([i["n_exponent_hits"] for i in infos])))
        n_sign = int(round(np.mean([i["n_sign_hits"] for i in infos])))
        return {
            "acc": float(acc),
            "crashed": int(crashed),
            "k": k_mean,
            "n_bits": n_bits,
            "n_exponent_hits": n_exp,
            "n_sign_hits": n_sign,
            "seconds": float(time.perf_counter() - t0),
            "last_finite_acc": float(acc),
            "ensemble_skipped": int(ensemble_skipped),
            "note": "; ".join(i["note"] for i in infos if i["note"]),
        }

    model = model_factory()
    info = _train_one_model(
        model=model,
        data=data,
        model_name=model_name,
        rad=rad,
        defense=defense_eff,
        seed=int(seed),
        rng=rng,
        r=r,
        T_days=T_days,
        p=p,
        flip_bias=flip_bias,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        momentum=momentum,
        allowed_bits=allowed_bits,
        single_hit=single_hit,
    )
    crashed = int(info["crashed"])
    if defense_eff == "clip":
        clip_all_params(model)
    acc = evaluate(model, data.x_test, data.y_test, model_name, clean_model=None)
    if acc is None:
        crashed = 1
        acc = info["last_finite_acc"] if info["last_finite_acc"] is not None else chance
    return {
        "acc": float(acc),
        "crashed": int(crashed),
        "k": float(info["k"]),
        "n_bits": int(info["n_bits"]),
        "n_exponent_hits": int(info["n_exponent_hits"]),
        "n_sign_hits": int(info["n_sign_hits"]),
        "seconds": float(time.perf_counter() - t0),
        "last_finite_acc": float(acc),
        "ensemble_skipped": int(ensemble_skipped),
        "note": info["note"],
    }
```

- 앙상블 학습은 복제본 3개를 **각각** 무방어로 학습시킨 뒤(각자 독립적으로 방사선을 맞음) 출력을 합친다. `crashed`는 "복제본 하나라도 죽었으면 1".
- CNN/ResNet은 앙상블 학습이 너무 무거워 `none`으로 대체하고 `ensemble_skipped=1`로 표시한다.

### 깨끗한 모델 만들기와 저장 (12주차 `SAVED_STATE`)

```python
# engine.py  (line 690~)
def ckpt_path(ckpt_dir: str, model_name: str, n_train: int | None = None, epochs: int | None = None) -> str:
    """Checkpoint name carries the training budget so a changed --n-train/--epochs
    never silently reuses an old file."""
    if n_train is None or epochs is None:
        return os.path.join(ckpt_dir, f"{model_name}_clean.pt")
    return os.path.join(ckpt_dir, f"{model_name}_clean_n{int(n_train)}_e{int(epochs)}.pt")
```

```python
# engine.py  (line 698~)
def train_clean_checkpoint(
    *,
    model_factory,
    data,
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    momentum: float,
    ckpt_dir: str,
    seed: int = 2026,
) -> tuple[dict, float, float]:
    """Train a clean model, save state_dict, return (state, acc, seconds)."""
    os.makedirs(ckpt_dir, exist_ok=True)
    path = ckpt_path(ckpt_dir, model_name, int(data.x_train.shape[0]), int(epochs))
    t0 = time.perf_counter()
    torch.manual_seed(int(seed))
    model = model_factory()
    opt = _make_opt(model, lr, momentum)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    n_train = int(data.x_train.shape[0])
    for _epoch in range(int(epochs)):
        model.train()
        perm = torch.randperm(n_train)
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            if idx.numel() < 2:
                continue
            opt.zero_grad(set_to_none=True)
            logits = model(data.x_train[idx])
            if model_name == "toy50":
                loss = bce(logits.reshape(-1), data.y_train[idx].float())
            else:
                loss = ce(logits, data.y_train[idx])
            loss.backward()
            opt.step()
    acc = evaluate(model, data.x_test, data.y_test, model_name, clean_model=None)
    if acc is None:
        acc = chance_acc_of(model_name)
    state = copy.deepcopy(model.state_dict())
    torch.save({"state_dict": state, "acc": acc, "model_name": model_name}, path)
    seconds = time.perf_counter() - t0
    print(f"[ckpt] {model_name} clean acc={acc:.2f}%  {seconds:.2f}s  → {path}")
    return state, float(acc), float(seconds)
```

```python
# engine.py  (line 745~)
def load_or_train_clean(
    *,
    model_factory,
    data,
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    momentum: float,
    ckpt_dir: str,
    seed: int = 2026,
    force: bool = False,
) -> tuple[dict, float]:
    path = ckpt_path(ckpt_dir, model_name, int(data.x_train.shape[0]), int(epochs))
    if (not force) and os.path.isfile(path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        print(f"[ckpt] loaded {path}  stored_acc={blob.get('acc', float('nan')):.2f}%")
        return blob["state_dict"], float(blob.get("acc", float("nan")))
    state, acc, _s = train_clean_checkpoint(
        model_factory=model_factory,
        data=data,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        momentum=momentum,
        ckpt_dir=ckpt_dir,
        seed=seed,
    )
    return state, acc
```

- 12주차의 `SAVED_STATE = copy.deepcopy(model.state_dict())`를 파일로 저장한 것. 파일 이름에 `n{학습데이터수}_e{epoch}`를 넣어, 설정을 바꾸면 새로 학습하도록 했다(점검에서 고침).

**생각해 볼 질문**
1. 방어 없는 stress 1% 학습은 파일럿에서 **100% 붕괴**했는데, 같은 조건의 추론은 12~13%였다(붕괴는 1/8). 학습이 더 잘 죽는 이유를 옵티마이저(momentum)로 설명해 보자.
2. 5중 검사 중 "그래디언트 검사"를 빼면 어떤 일이 생길 수 있나?

---

## 12단계. 실험 대상 — 모델 5개와 데이터

**무엇을 하나.** 수업의 곱셈판(toy50)부터 ResNet-18까지, 크기가 다른 모델 5개. "큰 모델일수록 N_bits가 커서 하루에 더 자주 맞는다"를 보기 위함.

```python
# models.py  (line 55~)
class Toy50(nn.Module):
    """Course toy: 50 float32 weights. Stored as (1, 50) so dim>=2 is flippable."""

    def __init__(self, std: float = 0.5) -> None:
        super().__init__()
        self.fc = nn.Linear(50, 1, bias=False)
        nn.init.normal_(self.fc.weight, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 50) → (B, 1)
        return self.fc(x)
```

```python
# models.py  (line 68~)
class MNISTLinear(nn.Module):
    """784-128-10 MLP."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.size(0), -1)
        return self.fc2(F.relu(self.fc1(x)))
```

```python
# models.py  (line 81~)
class MNISTCNN(nn.Module):
    """Small Conv-Conv-FC on 28×28 MNIST. ~220k weights."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))  # 14×14
        x = self.pool(F.relu(self.conv2(x)))  # 7×7
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

```python
# models.py  (line 244~)
def make_toy_data(
    n_train: int = 2000,
    n_test: int = 500,
    seed: int = 2026,
) -> TensorDataset:
    """Synthetic linear task. Teacher weights ~ N(0, 0.5^2); y = sign(W·x)."""
    rng = np.random.default_rng(int(seed))
    teacher = rng.normal(0.0, 0.5, size=(1, 50)).astype(np.float32)
    x_tr = rng.standard_normal((n_train, 50), dtype=np.float32)
    x_te = rng.standard_normal((n_test, 50), dtype=np.float32)
    y_tr = (np.sign(x_tr @ teacher.T).ravel() >= 0).astype(np.int64)
    y_te = (np.sign(x_te @ teacher.T).ravel() >= 0).astype(np.int64)
    return TensorDataset(
        torch.from_numpy(x_tr),
        torch.from_numpy(y_tr),
        torch.from_numpy(x_te),
        torch.from_numpy(y_te),
        extra={"teacher": teacher},
    )
```

```python
# models.py  (line 222~)
def load_mnist(n_train: int, n_test: int) -> TensorDataset:
    os.makedirs(DATA_DIR, exist_ok=True)
    train_ds = datasets.MNIST(root=DATA_DIR, train=True, download=True)
    test_ds = datasets.MNIST(root=DATA_DIR, train=False, download=True)
    x_train = train_ds.data[:n_train].unsqueeze(1).float() / 255.0  # (N,1,28,28)
    y_train = train_ds.targets[:n_train].long()
    x_test = test_ds.data[:n_test].unsqueeze(1).float() / 255.0
    y_test = test_ds.targets[:n_test].long()
    return TensorDataset(x_train, y_train, x_test, y_test)
```

```python
# models.py  (line 276~)
def default_budget(model_name: str) -> dict:
    """Starting train/test sizes and epochs; paper_sweep may shrink after a pilot."""
    name = model_name.lower().strip()
    if name == "toy50":
        return dict(n_train=2000, n_test=500, epochs=8, ckpt_epochs=8, batch_size=64, lr=0.05, momentum=0.0)
    if name == "mnist_linear":
        return dict(n_train=60_000, n_test=10_000, epochs=2, ckpt_epochs=2, batch_size=128, lr=0.05, momentum=0.9)
    if name == "mnist_cnn":
        return dict(n_train=60_000, n_test=10_000, epochs=2, ckpt_epochs=2, batch_size=128, lr=0.05, momentum=0.9)
    if name == "resnet18_t10":
        # train-mode radiation: 1 epoch (last layer). Infer clean ckpt: a few more epochs
        # of the cheap frozen-backbone head so baseline acc is not chance-level.
        return dict(n_train=10_000, n_test=2_000, epochs=1, ckpt_epochs=8, batch_size=64, lr=0.01, momentum=0.9)
    if name == "cifar_deep":
        return dict(n_train=10_000, n_test=2_000, epochs=1, ckpt_epochs=6, batch_size=64, lr=0.05, momentum=0.9)
    raise ValueError(name)
```

```python
# models.py  (line 189~)
def describe_model(model: nn.Module, flip_bias: bool = False) -> dict:
    params = collect_weight_params(model, flip_bias=flip_bias)
    n_w = n_weight_elements(params)
    n_b = n_bits_of(params)
    n_bias = sum(int(p.numel()) for n, p in model.named_parameters() if p.dim() == 1)
    n_all = sum(int(p.numel()) for p in model.parameters())
    return {
        "n_weight_elements": n_w,
        "n_bits": n_b,
        "n_bias_elements": n_bias,
        "n_all_params": n_all,
        "n_tensors": len(params),
    }
```

**줄별 해설**
- `Toy50` : 13주차의 "가중치 50개 곱셈판"을 `Linear(50, 1, bias=False)`로. `(1, 50)` 모양이라 dim ≥ 2 → 플립 대상. 초기값 정규분포 std 0.5도 수업과 같다.
- `make_toy_data` : 정답을 아는 "교사" 가중치를 하나 뽑고 `y = sign(교사·x)`로 라벨을 만든다. 학습은 이 교사를 흉내 내는 것.
- `MNISTLinear` 784→128→10 : 수업 SimpleNet(784→128→64→10)보다 층이 하나 적다. 논문에서 어느 쪽을 쓸지 결정 D2.
- `load_mnist` : 정규화 없이 /255만. 수업 코드는 평균·표준편차 정규화를 했다.
- `default_budget` : 모델마다 학습 데이터 수, epoch, 배치, 학습률. CPU에서 돌아가도록 CIFAR는 10,000장, 1 epoch.
- `describe_model` : N_bits와 편향 개수를 세어 로그·NOTES에 남긴다.

**생각해 볼 질문**
1. `Toy50`의 가중치를 `(50,)` 1차원으로 만들었다면 이 프로그램에서 어떤 일이 생기나? (2단계 `collect_weight_params` 참고)
2. mnist_cnn의 가중치 220,064개를 층별로 계산해 보자. (3×3 커널, 1→32→64 채널, FC 3136→64→10)

---

## 13단계. 실험 자동화 — 조종석

**무엇을 하나.** 13주차 실습 2-4의 "확률 5개 × 3회 반복 = 15번"을 일반화한 것. 셀(Cell) 하나 = 모델 × 모드 × 방사선 × 방어 × 반복 횟수. 시드는 셀과 회차에서 결정론적으로 만들어 CSV에 기록한다.

```python
# presets.py  (line 28~)
@dataclass
class Cell:
    model: str
    mode: str          # infer | train
    rad: str           # poisson | leo_saa | stress
    defense: str
    trials: int
    p: float = 0.01
    T_days: float = 1.0
    train_hours: float = 24.0
    epochs: int | None = None          # None → model default / pilot
    n_train: int | None = None
    n_test: int | None = None
    optional: bool = False             # may be dropped after pilot
    note: str = ""
    allowed_bits: str | None = None    # e.g. "30", "23-30", "exponent"; None = all 32 bits
    single_hit: bool = False           # exactly one event (bit-criticality experiment)
```

```python
# simulate.py  (line 75~)
def trial_seed(model: str, mode: str, rad: str, defense: str, trial: int, master: int = MASTER_SEED) -> int:
    """Distinct recorded seed per cell × trial. Derived from MASTER_SEED=2026.

    2026-09-11: stride per cell raised 20 → 1000 so up to 1000 trials per cell
    cannot collide with a neighbouring cell. Seeds therefore differ from the
    2026-08-19 pilot sweep (that sweep stays reproducible from its recorded seeds).
    """
    import hashlib
    key = f"{model}|{mode}|{rad}|{defense}".encode("utf-8")
    h = int(hashlib.md5(key).hexdigest()[:6], 16) % 100_000
    return int(master) * 100_000_000 + h * 1000 + int(trial)
```

```python
# simulate.py  (line 68~)
def parse_T(s: str) -> float:
    s = str(s).strip().lower()
    if s in ("saa_pass", "saa", "one_saa", "pass"):
        return float(SAA_PASS_DAYS)  # 12/1440
    return float(s)
```

```python
# simulate.py  (line 106~)
def run_cell(
    cell: Cell,
    *,
    r: float,
    flip_bias: bool,
    ckpt_dir: str,
    master_seed: int,
    force_ckpt: bool = False,
    data_cache: dict,
    ckpt_cache: dict,
    budget_overrides: dict,
) -> list[dict]:
    name = cell.model
    bud = dict(default_budget(name))
    bud.update({k: v for k, v in budget_overrides.get(name, {}).items() if v is not None})
    if cell.epochs is not None:
        bud["epochs"] = int(cell.epochs)
    if cell.n_train is not None:
        bud["n_train"] = int(cell.n_train)
    if cell.n_test is not None:
        bud["n_test"] = int(cell.n_test)

    cache_key = (name, bud["n_train"], bud["n_test"])
    if cache_key not in data_cache:
        print(f"[data] loading {name} n_train={bud['n_train']} n_test={bud['n_test']} ...")
        data_cache[cache_key] = load_data_for(name, bud["n_train"], bud["n_test"], toy_seed=master_seed)
    data = data_cache[cache_key]
    fac = factory_for(name)

    # N_bits from a fresh model (and print once)
    probe = fac()
    meta = describe_model(probe, flip_bias=flip_bias)
    n_bits = int(meta["n_bits"])
    print(
        f"[model] {name} n_weight_elements={meta['n_weight_elements']}  "
        f"N_bits={n_bits}  n_bias={meta['n_bias_elements']}  all_params={meta['n_all_params']}"
    )
    del probe

    T_days = float(cell.T_days)
    if cell.mode == "train" and cell.rad != "stress":
        # ASSUMPTION: map the whole training run to train_hours of satellite time.
        T_days = float(cell.train_hours) / 24.0

    rows: list[dict] = []
    clean_state = None
    clean_acc = float("nan")
    if cell.mode == "infer":
        ck = (name, bud["n_train"], int(bud.get("ckpt_epochs", bud["epochs"])))
        if ck not in ckpt_cache:
            state, acc = load_or_train_clean(
                model_factory=fac,
                data=data,
                model_name=name,
                epochs=int(bud.get("ckpt_epochs", bud["epochs"])),
                batch_size=int(bud["batch_size"]),
                lr=float(bud["lr"]),
                momentum=float(bud["momentum"]),
                ckpt_dir=ckpt_dir,
                seed=master_seed,
                force=force_ckpt,
            )
            ckpt_cache[ck] = (state, acc)
        clean_state, clean_acc = ckpt_cache[ck]
        # Reference measured with the same metric as the trials (toy50 → 100 = self-agreement).
        clean_acc = clean_reference_acc(fac, clean_state, data, name)

    n_trials = int(cell.trials)
    for trial in range(n_trials):
        seed = trial_seed(name, cell.mode, cell.rad, cell.defense, trial, master=master_seed)
        if cell.mode == "infer":
            out = run_infer_trial(
                model_factory=fac,
                clean_state=clean_state,
                data=data,
                model_name=name,
                rad=cell.rad,
                defense=cell.defense,
                seed=seed,
                r=r,
                T_days=T_days,
                p=cell.p,
                flip_bias=flip_bias,
                allowed_bits=cell.allowed_bits,
                single_hit=cell.single_hit,
            )
        else:
            out = run_train_trial(
                model_factory=fac,
                data=data,
                model_name=name,
                rad=cell.rad,
                defense=cell.defense,
                seed=seed,
                r=r,
                T_days=T_days,
                p=cell.p,
                flip_bias=flip_bias,
                epochs=int(bud["epochs"]),
                batch_size=int(bud["batch_size"]),
                lr=float(bud["lr"]),
                momentum=float(bud["momentum"]),
                allowed_bits=cell.allowed_bits,
                single_hit=cell.single_hit,
            )
        row = {
            "model": name,
            "mode": cell.mode,
            "rad": cell.rad,
            "defense": cell.defense,
            "trial": int(trial),
            "seed": int(seed),
            "acc": float(out["acc"]),
            "crashed": int(out["crashed"]),
            "k": float(out["k"]),
            "n_bits": int(out["n_bits"]),
            "n_exponent_hits": int(out["n_exponent_hits"]),
            "n_sign_hits": int(out["n_sign_hits"]),
            "T_days": float(T_days),
            "r": float(r),
            "p": float(cell.p) if cell.rad == "stress" else float("nan"),
            "train_hours": float(cell.train_hours) if cell.mode == "train" else float("nan"),
            "epochs": int(bud["epochs"]) if cell.mode == "train" else int(bud.get("ckpt_epochs", bud["epochs"])),
            "allowed_bits": "" if cell.allowed_bits is None else str(cell.allowed_bits),
            "single_hit": int(cell.single_hit),
            "n_train": int(bud["n_train"]),
            "n_test": int(data.x_test.shape[0]),
            "seconds": float(out["seconds"]),
            "clean_acc": float(clean_acc) if cell.mode == "infer" else float("nan"),
            "last_finite_acc": float(out["last_finite_acc"]),
            "ensemble_skipped": int(out.get("ensemble_skipped", 0)),
            "note": "; ".join(s for s in (cell.note, out.get("note", "")) if s),
        }
        rows.append(row)
        if trial == 0 or trial + 1 == n_trials or (trial + 1) % 5 == 0:
            print(
                f"  [{name} {cell.mode} {cell.rad} {cell.defense}  "
                f"{trial+1}/{n_trials}] acc={row['acc']:.2f} crash={row['crashed']} "
                f"k={row['k']:.1f} {row['seconds']:.2f}s"
            )
    return rows
```

**줄별 해설**
- `Cell` : 실험 조건 한 벌. `allowed_bits`, `single_hit`가 점검 후 추가된 필드.
- `trial_seed` : 셀 이름을 md5로 해시해 셀마다 다른 구간을 만들고 `+ trial`. 같은 셀·같은 회차는 언제 돌려도 같은 시드 → **재현 가능**. 셀 간 간격을 20 → 1000으로 넓혀 반복 30회에도 겹치지 않게 했다.
- `run_cell` : 예산 결정 → 데이터 캐시 → N_bits 출력 → (추론이면) 클린 체크포인트 → 회차 반복 → CSV 행. `T_days`는 학습·비-stress일 때만 `train_hours/24`로 바꾼다.
- CSV 한 행의 열 이름을 13주차 DataFrame과 비교해 보자: `확률(%)`→`p`, `맞은_가중치`→`k`, `NaN_∞_개수`→`crashed`, `정확도(%)`→`acc`.

### 요약: 평균 대신 중앙값·생존률

```python
# simulate.py  (line 255~)
def add_survival_column(df: pd.DataFrame) -> pd.DataFrame:
    """survived = not crashed and acc >= SURVIVAL_FRACTION * clean reference.

    Outcomes are bimodal (fine or dead), so mean ± std is misleading; report
    survival_rate and median instead. Train rows have no clean_acc: borrow the
    model's infer reference if the sweep contains one, else NaN.
    """
    df = df.copy()
    ref = pd.to_numeric(df.get("clean_acc"), errors="coerce")
    if "mode" in df:
        infer_ref = df[df["mode"] == "infer"].groupby("model")["clean_acc"].first()
        ref = ref.where(ref.notna(), df["model"].map(infer_ref))
    ok = (df["crashed"] == 0) & (df["acc"] >= SURVIVAL_FRACTION * ref)
    df["survived"] = ok.astype(float).where(ref.notna(), np.nan)
    return df
```

```python
# simulate.py  (line 272~)
def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "mode", "rad", "defense"]
    if "single_hit" in df:
        keys += ["allowed_bits", "single_hit"]
    df = add_survival_column(df)
    rows = []
    for key, g in df.groupby(keys, sort=True, dropna=False):
        acc = g["acc"]
        q25, q75 = float(acc.quantile(0.25)), float(acc.quantile(0.75))
        row = dict(zip(keys, key))
        row.update(
            {
                "n": int(len(g)),
                "mean_acc": float(acc.mean()),
                "std_acc": float(acc.std(ddof=1)) if len(g) > 1 else 0.0,
                "median_acc": float(acc.median()),
                "iqr_acc": q75 - q25,
                "crash_rate": float(g["crashed"].mean()),
                "survival_rate": float(g["survived"].mean()) if g["survived"].notna().any() else float("nan"),
                "mean_k": float(g["k"].mean()),
                "n_bits": int(g["n_bits"].iloc[0]),
                "mean_seconds": float(g["seconds"].mean()),
                "T_days": float(g["T_days"].mean()),
                "p": float(g["p"].mean()) if "p" in g else float("nan"),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
```

- `survived = (crashed == 0) and (acc ≥ 0.9 × clean_acc)`. 결과가 "멀쩡하거나 죽거나"라서 평균±표준편차는 오해를 낳는다(파일럿 mnist_cnn: 평균 86.4 ± 27.7 vs 생존률 0.8, 중앙값 98.8).
- 13주차 `groupby("확률(%)").agg(...)`의 확장판. 그룹 키에 `allowed_bits`, `single_hit`가 들어가 E2 결과도 한 표에 담긴다.

### 명령줄 옵션

```python
# simulate.py  (line 558~)
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multi-model LEO bit-flip SEU suite")
    p.add_argument("--model", choices=MODEL_NAMES, default="toy50")
    p.add_argument("--mode", choices=("infer", "train"), default="infer")
    p.add_argument("--rad", choices=("poisson", "leo_saa", "stress"), default="leo_saa")
    p.add_argument("--defense", choices=DEFENSE_NAMES, default="none")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--r", type=float, default=DEFAULT_R, help="SEU/bit/day (ignored by stress)")
    p.add_argument("--T", type=str, default="1.0", help="Exposure in days, or 'saa_pass' (=12/1440)")
    p.add_argument("--p", type=float, default=DEFAULT_STRESS_P, help="stress per-weight hit probability")
    p.add_argument("--train-hours", type=float, default=24.0,
                   help="ASSUMPTION: map the training run to this many hours of satellite time (24 → 1 LEO day)")
    p.add_argument("--flip-bias", action="store_true", help="Also flip dim-1 bias tensors")
    p.add_argument("--allowed-bits", type=str, default=None,
                   help="Restrict flips to these bit positions: '30', '23-30', '31,30', or sign|exponent|mantissa")
    p.add_argument("--single-hit", action="store_true",
                   help="Exactly ONE (weight, bit) event per trial (bit-criticality experiment); ignores r/T/p")
    p.add_argument("--tag", type=str, default="", help="Extra tag for the output CSV name")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--n-test", type=int, default=None)
    p.add_argument("--preset", choices=("paper_sweep",), default=None)
    p.add_argument("--out-dir", type=str, default=os.path.join(HERE, "results"))
    p.add_argument("--ckpt-dir", type=str, default=CKPT_DIR_DEFAULT)
    p.add_argument("--seed-master", type=int, default=MASTER_SEED)
    p.add_argument("--force-ckpt", action="store_true")
    return p
```

```python
# simulate.py  (line 739~)
def run_single(args) -> None:
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    self_test_bitflip()
    print("[self-test] bit-flip OK")
    T_days = parse_T(args.T)
    cell = Cell(
        model=args.model,
        mode=args.mode,
        rad=args.rad,
        defense=args.defense,
        trials=int(args.trials),
        p=float(args.p),
        T_days=T_days,
        train_hours=float(args.train_hours),
        epochs=args.epochs,
        n_train=args.n_train,
        n_test=args.n_test,
        allowed_bits=(None if args.allowed_bits is None else ",".join(str(b) for b in parse_allowed_bits(args.allowed_bits))),
        single_hit=bool(args.single_hit),
    )
    rows = run_cell(
        cell,
        r=args.r,
        flip_bias=args.flip_bias,
        ckpt_dir=os.path.abspath(args.ckpt_dir),
        master_seed=args.seed_master,
        force_ckpt=args.force_ckpt,
        data_cache={},
        ckpt_cache={},
        budget_overrides={},
    )
    df = pd.DataFrame(rows)
    tag = f"{args.model}_{args.mode}_{args.rad}_{args.defense}"
    if args.single_hit:
        tag += "_1hit"
    elif args.rad == "stress":
        tag += f"_p{args.p:g}"
    else:
        tag += f"_T{T_days:g}_r{args.r:g}"
    if args.allowed_bits:
        tag += "_bits" + str(args.allowed_bits).replace(",", "+")
    if args.tag:
        tag += "_" + args.tag
    path = os.path.join(out_dir, f"single_{tag}.csv")
    df.to_csv(path, index=False)
    print(f"[write] {path}")
    print(df[["trial", "seed", "acc", "crashed", "k", "n_bits", "seconds"]].to_string(index=False))
    summ = make_summary(df).iloc[0]
    print(
        f"mean_acc={summ['mean_acc']:.2f} ± {summ['std_acc']:.2f}  median={summ['median_acc']:.2f}  "
        f"survival={summ['survival_rate']:.2f}  crash={summ['crash_rate']:.2f}  "
        f"mean_k={summ['mean_k']:.2f}  N_bits={int(summ['n_bits'])}  clean_ref={df['clean_acc'].iloc[0]:.2f}"
    )
```

- `--single-hit --allowed-bits 30` : 30번 비트 하나만, 정확히 한 번(E2).
- `--defense ensemble_median` : 강건한 앙상블.
- 출력 파일 이름에 p/T/r/bits가 붙어 p를 바꿔 돌려도 덮어쓰지 않는다.

**생각해 볼 질문**
1. `trial_seed("toy50","infer","stress","none", 0)`과 `(…, "clip", 0)`의 시드가 다르다. 두 방어를 **같은 명중 위치**에서 비교하고 싶다면 코드를 어떻게 바꾸겠는가? (힌트: 키에서 defense를 빼면?) 그러면 무엇이 좋고 무엇이 나쁜가?
2. 생존 기준 0.9를 0.95로 바꾸면 결론이 달라질 수 있는 셀은 어떤 것들인가?

---

## 14. 직접 돌려 보기 & 리뷰 미션

### 실행 준비 (Windows, CPU)
```
python -m venv seuvenv
seuvenv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
seuvenv\Scripts\pip install numpy pandas matplotlib
cd sim\seu_suite
```

### 미션 1 — 자가진단 통과시키기
```
..\..\seuvenv\Scripts\python simulate.py --model toy50 --trials 1
```
첫 줄에 `[self-test] bit-flip OK`가 나와야 한다. 1단계 `self_test_bitflip`이 무엇을 검사했는지 말해 보자.

### 미션 2 — 비트 하나의 운명 (E2 맛보기)
```
python simulate.py --model mnist_linear --mode infer --rad stress --single-hit --allowed-bits 30 --trials 20
python simulate.py --model mnist_linear --mode infer --rad stress --single-hit --allowed-bits 5  --trials 20
```
점검 실측: 30번 비트 한 방은 중앙값 72.7점·생존률 50%, 5번 비트는 87.35점(무피해)·생존률 100%. 여러분 결과의 `survival_rate`를 기록하고, 두 비트가 float32의 어느 영역인지 말해 보자.

### 미션 3 — 평균 vs 중앙값 (10-4)
```
python simulate.py --model toy50 --mode infer --rad stress --p 0.05 --defense ensemble        --trials 20
python simulate.py --model toy50 --mode infer --rad stress --p 0.05 --defense ensemble_median --trials 20
```
`survival_rate` 차이를 적고, 왜 `nan_to_num`만으로는 부족했는지 한 문장으로.

### 미션 4 — 궤도 30일
```
python simulate.py --model mnist_linear --mode infer --rad leo_saa --T 30 --defense none --trials 10
python simulate.py --model mnist_linear --mode infer --rad leo_saa --T 30 --defense clip --trials 10
```
`mean_k`(30일 동안 명중 수)가 r·N_bits·30 ≈ 98에 가까운지 확인. 생존률은?

### 미션 5 — 코드 고치기
`combine_logits`에 `"trimmed"`(최대·최소 하나씩 버리고 평균)를 추가하고 `--defense ensemble_trimmed`로 돌릴 수 있게 만들어 보자. 바꿔야 할 파일은 몇 개인가? (힌트: `DEFENSE_NAMES`, `ENSEMBLE_DEFENSES`, `ensemble_reduce_of`)

### 미션 6 — 한계 찾기
이 프로그램이 진짜 우주와 다른 점을 코드에서 **세 군데** 찾아 줄 번호와 함께 적어 보자. (예: 편향 제외, 시간 매핑 가정, 단일 비트, r 가정, 온도·차폐 미고려…)

### 코드 읽기 퀴즈 (교사용 정답 포함)

| # | 질문 | 정답 |
|---|---|---|
| 1 | `xor_one_bit`에서 `np.searchsorted(cum, idx, side="right") - 1`이 하는 일은? | 전역 번호가 속한 텐서 번호 찾기 |
| 2 | `collect_weight_params`가 편향을 빼는 조건은? | `p.dim() >= 2`만 통과 |
| 3 | `R_SAA_MULTIPLIER`가 36인 이유는? | 0.90 ÷ (3×12/1440 = 0.025) |
| 4 | `events_by_step`에서 t=T인 이벤트는 몇 번째 스텝? | 마지막 스텝(n_steps−1) |
| 5 | `evaluate`가 `None`을 돌려주는 경우 두 가지는? | 가중치 비유한 / 출력 비유한 (+예외) |
| 6 | `write_median`이 중앙값 전에 `nan_to_num`을 하는 이유는? | NaN은 크기 비교가 안 되고, ∞가 두 복제본에 있으면 중앙값도 ∞가 되므로 먼저 유한값(0)으로 바꾼다 |
| 7 | `even_parity_u32`의 마지막 `& 1`은 무엇을 뜻하나? | popcount의 홀짝(짝수 패리티 비트) |
| 8 | `ensemble`과 `ensemble_median`의 코드 차이는 정확히 어디인가? | `combine_logits(stack, how)`의 `how` 한 단어 |
| 9 | 붕괴한 시행의 `acc`에 들어가는 값은? | `chance_acc_of(model)` = 10 또는 50 (학습은 마지막 유한 정확도가 있으면 그 값) |
| 10 | `trial_seed`가 셀마다 1000칸씩 띄우는 이유는? | 반복이 20회를 넘어도 옆 셀과 시드가 겹치지 않게 |
