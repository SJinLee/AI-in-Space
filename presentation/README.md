# presentation/ — 10분 발표 자료 (중학생 2인)

| 파일 | 내용 |
|---|---|
| `slides.pptx` / `slides.pdf` | 발표 슬라이드 11장 (16:9). 슬라이드 노트에 담당 발표자 표시 |
| `발표대본.md` | 슬라이드별 대본 (이태린 1~4·10~11, 최재원 5~9), 시간 배분, 예상 질문 6개 |
| `poster.pptx` / `poster.pdf` | A1 세로(594×841 mm) 포스터. 인쇄는 `poster.pdf` 사용 |
| `포스터대본.md` | 포스터 앞 5분 발표 대본 (결론 먼저 → 결과 → 공식, 포스터 번호 ①~⑧ 순서로 가리킴), 3분판·30초판, 추가 예상 질문 |
| `build_slides.js`, `build_poster.js` | pptxgenjs 생성 스크립트. 그림은 `../paper/figures/*.png`를 읽음 |
| (그림) | 이봉 분포는 `../paper/figures/fig_bimodal_a.png`(p 단계마다 100회, `src/results/bimodal100`) 사용 |

다시 만들기 (pptxgenjs 가 설치된 폴더에서):

```
set NODE_PATH=<pptxgenjs 가 있는 node_modules>
node presentation/build_slides.js presentation/slides.pptx
node presentation/build_poster.js presentation/poster.pptx
```

수치는 논문(`../paper/main.tex`, `../paper/tables/numbers.tex`) 2026-11-06 기준.
