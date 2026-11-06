# presentation/ — 10분 발표 자료 (중학생 2인)

| 파일 | 내용 |
|---|---|
| `slides.pptx` / `slides.pdf` | 발표 슬라이드 11장 (16:9). 슬라이드 노트에 담당 발표자 표시 |
| `발표대본.md` | 슬라이드별 대본 (이태린 1~4·10~11, 최재원 5~9), 시간 배분, 예상 질문 6개 |
| `poster.pptx` / `poster.pdf` | A1 세로(594×841 mm) 포스터. 인쇄는 `poster.pdf` 사용 |
| `build_slides.js`, `build_poster.js` | pptxgenjs 생성 스크립트. 그림은 `../paper/figures/*.png`를 읽음 |
| `fig_bimodal_a.png` | `fig_bimodal.png`의 (a) 패널만 잘라 낸 것 |

다시 만들기 (pptxgenjs 가 설치된 폴더에서):

```
set NODE_PATH=<pptxgenjs 가 있는 node_modules>
node presentation/build_slides.js presentation/slides.pptx
node presentation/build_poster.js presentation/poster.pptx
```

수치는 논문(`../paper/main.tex`, `../paper/tables/numbers.tex`) 2026-11-06 기준.
