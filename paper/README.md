# paper/ — 논문 원고

```
paper/
├─ main.tex           본문 (XeLaTeX + kotex, 한글)
├─ refs.bib           참고문헌
├─ make_figures.py    src/results/*.csv → figures/, tables/ 자동 생성
├─ figures/           fig_e1_*.pdf / .png
└─ tables/            table_e1.tex (표 2), numbers.tex (본문 수치 매크로)
```

## 컴파일

**Overleaf (권장)** — `paper/` 폴더 전체를 zip으로 올리고 *Menu → Compiler → XeLaTeX* 로 바꾼 뒤 Recompile.
한글 글꼴 `Noto Serif CJK KR` / `Noto Sans CJK KR`은 Overleaf에 기본 설치돼 있다.

**로컬 (MiKTeX, 2026-10-16 확인)** — 이 PC에는 MiKTeX 25.12가 `%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64` 에 설치돼 있고,
누락 패키지 자동 설치(`[MPM]AutoInstall=1`)를 켜 두었다. PowerShell에서:

```
cd paper
& "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64\latexmk.exe" -xelatex main.tex
```

한글 글꼴은 `main.tex` 상단에서 Noto Serif CJK KR → 나눔명조 → 맑은 고딕 순으로 있는 것을 자동 선택한다
(Overleaf는 Noto, 이 PC는 맑은 고딕). 중간 산출물(`main.aux`, `main.xdv`, `build.log` 등)은 지워도 된다.

## 그림·표·수치 다시 만들기

```
python paper/make_figures.py
```

`src/results/single_mnist_linear_infer_stress_*.csv`를 읽어 그림 3장, 표 1개, 그리고 본문이 쓰는
수치 매크로(`\CleanAcc`, `\PhalfNone` 등)를 다시 계산한다. 본문의 숫자는 손으로 고치지 말고
CSV를 바꾼 뒤 이 스크립트를 다시 돌린다.

## 실험 데이터

- E1 (mnist_linear, p 13단계 × none/clip × 10회): `src/results/single_mnist_linear_infer_stress_*.csv`
- E1-M 모델 크기 축 (mnist_cnn, cifar_deep, resnet18_t10): `python src/run_e1_models.py --models ... --trials 10 --skip-existing`
  (2026-10-16 실행, CPU 기준 mnist_cnn 2분, cifar_deep 6분, resnet18 9분. CIFAR-10 170MB와 ResNet-18 가중치 45MB를 처음 한 번 내려받음)

- E3 궤도 모드: `python src/run_e1_models.py --rad leo_saa --models mnist_linear mnist_cnn --T-list 1 3 7 14 30 60 90 --trials 10`
  (resnet18 은 `--T-list 0.03 0.1 0.3 1 3`)
- E5 방어 6종: `python src/run_e1_models.py --models mnist_linear --p-list 3e-4 1e-3 3e-3 1e-2 3e-2 --defenses none clip tmr ensemble ensemble_median parity --trials 10`
- E2 비트별 치명도: `python src/run_e1_models.py --models mnist_linear --bits 31 30 29 28 27 26 25 24 23 22 16 10 0 --defenses none clip --trials 50`
  (다른 모델은 `--defenses none --trials 30`)
- E5 고선량(TMR 한계 검증): `python src/run_e1_models.py --models mnist_linear --p-list 0.05 0.1 0.2 0.3 0.5 --defenses tmr parity --trials 10`
- E4 학습 중 주입: `python src/run_e1_models.py --models mnist_linear --mode train --p-list 0 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 --defenses none clip parity tmr --trials 10`
  (궤도 모드: `--mode train --rad leo_saa --train-hours 24 168 720 2160 --defenses none clip`; mnist_cnn 은 `--p-list 0 1e-3 1e-2 --defenses none clip --trials 5`, 셀당 약 5분)
- 수학 모델(모델 A 지수, 모델 B 로그-로지스틱)은 `paper/model_fit.py` 에 있고 `make_figures.py` 가 stress 데이터로 맞춘 뒤 E3 로 검증한다.

## 체크포인트

추론 실험은 `src/results/ckpts/*.pt`(방사선 전의 깨끗한 학습 결과)에서 출발한다. resnet18 체크포인트(43 MB)만
git에 없으며, 재생성 방법은 `src/README.md` 7-1절에 있다 (첫 실행 시 자동 생성).

## 아직 비어 있는 곳

`main.tex`에서 노란 상자(`\todo{...}`)로 표시한 부분은 데이터가 없어 채우지 못한 곳이다.
`src/seu_suite_colab.ipynb`의 3-2 ~ 3-5절 셀(E2 비트별 치명도, E3 궤도 모드, E4 학습 중 주입,
E5 방어 비교)을 실행해 CSV를 `src/results/`에 넣은 뒤 `make_figures.py`에 집계를 추가하면 된다.
