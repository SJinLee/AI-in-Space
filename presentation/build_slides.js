// 10분 발표용 슬라이드 (중학생 2인) — pptxgenjs
const pptxgen = require("pptxgenjs");
const path = require("path");
const { applyTheme } = require("C:/Users/math/.claude/skills/synced/c3517a04-a7ed-4c5f-aafd-36f5f56c8606_019c1aca-84c5-42ca-9ebf-35917cb803a7/pptx/scripts/apply_theme.js");

const FIG = "C:/Users/math/Dropbox/daejin(me)/영재/2026/AI-in-space/paper/figures/";
const OUT = process.argv[2];

const THEME = {
  name: "SEU Space",
  headFontFace: "Malgun Gothic",
  bodyFontFace: "Malgun Gothic",
  colors: {
    dk1: "0B1B3A", lt1: "FFFFFF", dk2: "4A566E", lt2: "EEF2F8",
    accent1: "D9532B", accent2: "2E6DB4", accent3: "F2B134",
    accent4: "2A9D8F", accent5: "8A94A6", accent6: "6C3FB5",
    hlink: "2E6DB4", folHlink: "6C3FB5",
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "우주에서 AI는 살아남을까?";
pres.author = "이태린, 최재원";
const C = pres.SchemeColor;
const W = 13.33, H = 7.5;

// 별 (고정 좌표, 재현 가능)
function stars(n, seed) {
  const out = [];
  let s = seed;
  const rnd = () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
  for (let i = 0; i < n; i++) {
    const d = 0.04 + rnd() * 0.07;
    out.push({ ellipse: { x: rnd() * (W - 0.2), y: rnd() * (H - 0.2), w: d, h: d, fill: { color: C.background1, transparency: Math.round(rnd() * 50) } } });
  }
  return out;
}

pres.defineSlideMaster({
  title: "DARK",
  background: { color: THEME.colors.dk1 },
  objects: [
    ...stars(70, 7),
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.2, w: 11.7, h: 1.6, fontSize: 40, bold: true, color: C.background1, align: "left", valign: "bottom", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.95, w: 11.7, h: 2.2, fontSize: 18, color: C.background2, align: "left", valign: "top", margin: 0 }, text: "" } },
  ],
});

pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: THEME.colors.lt1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.9, fontSize: 30, bold: true, color: C.text1, align: "left", valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "우주 방사선과 인공신경망  ·  이태린 · 최재원", options: { x: 0.6, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: C.accent5, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
});

// ---------- helpers ----------
let n = 0;
function textbox(slide, text, o) {
  n++;
  return slide.addText(text, Object.assign({ isTextBox: true, margin: 0, fontSize: 16, color: C.text1, objectName: "text" + n }, o));
}
function card(slide, x, y, w, h, opts = {}) {
  n++;
  slide.addShape(pres.ShapeType.roundRect, Object.assign({ x, y, w, h, rectRadius: 0.12, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "card" + n }, opts));
}
function stat(slide, x, y, w, big, label, color, bigSize = 36) {
  card(slide, x, y, w, 1.55);
  textbox(slide, big, { x: x + 0.2, y: y + 0.12, w: w - 0.4, h: 0.85, fontSize: bigSize, bold: true, color, align: "center", valign: "middle" });
  textbox(slide, label, { x: x + 0.2, y: y + 0.98, w: w - 0.4, h: 0.5, fontSize: 12, color: C.text2, align: "center", valign: "top" });
}
function bullets(slide, items, o) {
  const arr = items.map((t, i) => ({ text: t, options: { bullet: { indent: 14 }, breakLine: i < items.length - 1, paraSpaceAfter: 8 } }));
  return textbox(slide, arr, Object.assign({ fontSize: 16, valign: "top" }, o));
}
function numCircle(slide, x, y, num, color) {
  n++;
  slide.addShape(pres.ShapeType.ellipse, { x, y, w: 0.5, h: 0.5, fill: { color }, line: { color }, objectName: "num" + n });
  textbox(slide, String(num), { x, y, w: 0.5, h: 0.5, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle" });
}
function arrow(slide, x, y, w) {
  n++;
  slide.addShape(pres.ShapeType.rightArrow, { x, y, w, h: 0.45, fill: { color: C.accent5 }, line: { color: C.accent5 }, objectName: "arrow" + n });
}

// ======================= 1. 표지 =======================
pres.addSection({ title: "도입" });
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: "도입" });
  textbox(s, "2026 영재교육 · AI in Space", { x: 0.8, y: 1.5, w: 8, h: 0.4, fontSize: 14, color: C.accent3, bold: true });
  s.addText("우주에서 AI는 살아남을까?", { placeholder: "title" });
  s.addText([
    { text: "방사선이 뒤집은 비트 하나가 인공신경망에 미치는 영향", options: { fontSize: 20, color: C.background1, breakLine: true } },
    { text: " ", options: { fontSize: 10, breakLine: true } },
    { text: "이태린 (민락중학교)  ·  최재원 (선덕중학교)", options: { fontSize: 16, breakLine: true } },
    { text: "지도교사 이성진 · 조교 이유림 (대진대학교)", options: { fontSize: 13, color: C.accent5 } },
  ], { placeholder: "body" });
  // 위성 아이콘 느낌의 간단한 도형
  n++; s.addShape(pres.ShapeType.ellipse, { x: 10.3, y: 4.4, w: 2.0, h: 2.0, fill: { color: C.accent2 }, line: { color: C.accent2 }, objectName: "earth" });
  n++; s.addShape(pres.ShapeType.ellipse, { x: 9.6, y: 3.9, w: 3.4, h: 3.0, fill: { type: "none" }, line: { color: C.accent3, width: 1.5, dashType: "dash" }, objectName: "orbit" });
  n++; s.addShape(pres.ShapeType.rect, { x: 12.75, y: 5.1, w: 0.22, h: 0.22, fill: { color: C.accent3 }, line: { color: C.accent3 }, objectName: "sat" });
  s.addNotes("표지. 발표자 1(이태린)이 인사하고 제목을 읽는다.");
}

// ======================= 2. 배경 =======================
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "도입" });
  s.addText("위성 속 AI에는 '비트 하나가 뒤집히는' 사고가 매일 일어난다", { placeholder: "title" });
  const items = [
    ["☀", "우주 방사선", "고에너지 양성자·중이온이 메모리 셀에 전하를 남겨 0↔1을 바꾼다. 이것을 단일 사건 오류(SEU)라고 한다.", C.accent1],
    ["🛰", "위성에 AI 탑재", "위성 영상 판독, 자율 항법 등을 위해 신경망을 위성 안에 직접 넣으려는 시도가 늘고 있다.", C.accent2],
    ["?", "우리의 질문", "신경망 가중치의 비트 하나가 뒤집히면 무슨 일이 생길까? 얼마나 버틸까? 어떻게 막을까?", C.accent4],
  ];
  items.forEach(([ico, head, body, col], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 1.7, 3.85, 3.6);
    n++; s.addShape(pres.ShapeType.ellipse, { x: x + 0.3, y: 2.0, w: 0.8, h: 0.8, fill: { color: col }, line: { color: col }, objectName: "ico" + i });
    textbox(s, ico, { x: x + 0.3, y: 2.0, w: 0.8, h: 0.8, fontSize: 22, bold: true, color: C.background1, align: "center", valign: "middle" });
    textbox(s, head, { x: x + 0.3, y: 3.0, w: 3.3, h: 0.5, fontSize: 20, bold: true });
    textbox(s, body, { x: x + 0.3, y: 3.55, w: 3.25, h: 1.6, fontSize: 14, color: C.text2, valign: "top" });
  });
  textbox(s, [
    { text: "상용 메모리 1 GB 는 저궤도에서 하루 약 8,600번 비트가 뒤집힌다 (비트 하나가 하루에 뒤집힐 확률 r = 10" },
    { text: "−6", options: { superscript: true } },
    { text: "). 특히 남대서양 이상 지역(SAA)을 지날 때 몰려서 일어난다." },
  ], { x: 0.6, y: 5.6, w: 12.1, h: 1.0, fontSize: 15, color: C.text1, fill: { color: "FFF4E8" }, margin: 12, valign: "middle" });
  s.addNotes("발표자 1. 배경 설명.");
}

// ======================= 3. float32 =======================
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "도입" });
  s.addText("가중치 하나 = 32비트. 어느 자리가 뒤집히느냐가 전부 다르다", { placeholder: "title" });
  // 32-bit bar
  const bx = 0.6, by = 1.65, bh = 0.7, unit = 12.1 / 32;
  const segs = [[1, C.accent2, "부호 (1비트)", "bit 31"], [8, C.accent1, "지수 (8비트)", "bit 30–23"], [23, C.accent5, "가수 (23비트)", "bit 22–0"]];
  let cx = bx;
  segs.forEach(([bits, col, lab, rng], i) => {
    n++; s.addShape(pres.ShapeType.rect, { x: cx, y: by, w: unit * bits, h: bh, fill: { color: col }, line: { color: C.background1, width: 1 }, objectName: "seg" + i });
    if (bits > 1) textbox(s, lab, { x: cx, y: by, w: unit * bits, h: bh, fontSize: 14, bold: true, color: C.background1, align: "center", valign: "middle" });
    textbox(s, rng, { x: cx, y: by + bh + 0.05, w: Math.max(unit * bits, 1.0), h: 0.3, fontSize: 11, color: C.text2, align: bits === 1 ? "left" : "center" });
    cx += unit * bits;
  });
  textbox(s, "값 = (−1)^부호 × 2^(지수−127) × 1.가수      예) 1.0 = 0 01111111 000…0", { x: 0.6, y: 2.75, w: 12.1, h: 0.4, fontSize: 14, color: C.text2 });
  // examples
  const ex = [
    ["bit 0 (가수 끝)", "1.0 → 1.0000001", "거의 변화 없음", C.accent5],
    ["bit 22 (가수 맨 앞)", "1.0 → 1.5", "조금 커짐 · 무해", C.accent5],
    ["bit 31 (부호)", "1.0 → −1.0", "부호 반전 · 크기는 그대로", C.accent2],
    ["bit 30 (지수 맨 앞)", "1.0 → +∞  (0.99 → 3×10³⁸)", "값이 10³⁸ 배 · 치명적", C.accent1],
  ];
  ex.forEach(([a, b, c, col], i) => {
    const x = 0.6 + i * 3.05;
    card(s, x, 3.4, 2.9, 2.3);
    n++; s.addShape(pres.ShapeType.rect, { x: x + 0.25, y: 3.65, w: 0.5, h: 0.5, fill: { color: col }, line: { color: col }, objectName: "exico" + i });
    textbox(s, a, { x: x + 0.9, y: 3.65, w: 1.95, h: 0.5, fontSize: 13, bold: true, valign: "middle" });
    textbox(s, b, { x: x + 0.25, y: 4.3, w: 2.45, h: 0.6, fontSize: 16, bold: true, color: col, valign: "middle" });
    textbox(s, c, { x: x + 0.25, y: 4.95, w: 2.45, h: 0.6, fontSize: 13, color: C.text2, valign: "top" });
  });
  textbox(s, "지수 비트 8개 중 맨 앞 하나(bit 30)가 0→1이 되면 2¹²⁸ 배가 되어, 한 층만 지나도 출력 전체를 집어삼킨다.",
    { x: 0.6, y: 5.95, w: 12.1, h: 0.75, fontSize: 15, bold: true, color: C.accent1, valign: "middle" });
  s.addNotes("발표자 1. float32 구조.");
}

// ======================= 4. 방법 =======================
pres.addSection({ title: "방법" });
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "방법" });
  s.addText("실험 방법: 비트 하나를 진짜로 XOR 하는 시뮬레이터", { placeholder: "title" });
  const steps = [
    ["학습 끝난 모델", "MNIST 손글씨 분류\n가중치 101,632개\n= 3,252,224비트", C.accent2],
    ["가중치 뽑기", "각 가중치를 확률 p로 선택\np = 0.001 % ~ 10 %\n(13단계)", C.accent4],
    ["비트 하나 XOR", "32비트 중 무작위 한 자리\n0↔1 (가우시안 잡음 아님)", C.accent1],
    ["정확도 측정", "시험 10,000장으로 채점\n각 p마다 10회 반복\n총 260회", C.accent3],
  ];
  steps.forEach(([h, b, col], i) => {
    const x = 0.6 + i * 3.15;
    card(s, x, 1.65, 2.75, 2.6);
    numCircle(s, x + 0.25, 1.9, i + 1, col);
    textbox(s, h, { x: x + 0.9, y: 1.9, w: 1.8, h: 0.5, fontSize: 16, bold: true, valign: "middle" });
    textbox(s, b, { x: x + 0.25, y: 2.55, w: 2.3, h: 1.6, fontSize: 13, color: C.text2, valign: "top" });
    if (i < 3) arrow(s, x + 2.78, 2.72, 0.34);
  });
  // 비교 조건
  textbox(s, "비교한 조건", { x: 0.6, y: 4.5, w: 4, h: 0.4, fontSize: 16, bold: true });
  card(s, 0.6, 4.95, 5.9, 1.75);
  textbox(s, [
    { text: "방어 없음 (none)", options: { bold: true, color: C.accent1, breakLine: true } },
    { text: "뒤집힌 그대로 사용한다.", options: { color: C.text2, breakLine: true } },
    { text: "클리핑 (clip)", options: { bold: true, color: C.accent2, breakLine: true } },
    { text: "NaN·∞ 를 0으로 바꾸고, 값을 [−3, 3] 안으로 자른다. 가장 단순한 방어.", options: { color: C.text2 } },
  ], { x: 0.85, y: 5.1, w: 5.4, h: 1.5, fontSize: 14, valign: "top" });
  textbox(s, "확장 실험", { x: 6.8, y: 4.5, w: 4, h: 0.4, fontSize: 16, bold: true });
  card(s, 6.8, 4.95, 5.9, 1.75);
  bullets(s, [
    "모델 4종 (가중치 10만 ~ 1,100만 개, ResNet-18 포함)",
    "비트 자리별로 딱 하나만 뒤집기 (E2)",
    "SAA 통과를 흉내 낸 궤도 모드 (E3)",
    "방어 6종 비교: 클리핑·패리티·TMR·앙상블 (E5)",
  ], { x: 7.05, y: 5.1, w: 5.4, h: 1.5, fontSize: 13, color: C.text2 });
  s.addNotes("발표자 1. 실험 방법. 끝나면 발표자 2에게 넘긴다.");
}

// ======================= 5. 결과 1: p에 따른 정확도 =======================
pres.addSection({ title: "결과" });
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "결과" });
  s.addText("결과 ① 방어가 없으면 17일, 클리핑만 켜도 1,754일", { placeholder: "title" });
  s.addImage({ path: FIG + "fig_e1_dose_response.png", x: 0.6, y: 1.45, w: 8.3, h: 5.19, objectName: "그림: p에 따른 정확도" });
  stat(s, 9.3, 1.45, 3.4, "17일", "방어 없음, 생존률 50 % 노출 일수\n(p ≈ 0.055 %)", C.accent1);
  stat(s, 9.3, 3.2, 3.4, "1,754일", "클리핑, 생존률 50 % 노출 일수\n(p ≈ 5.6 %)", C.accent2);
  stat(s, 9.3, 4.95, 3.4, "×102", "클리핑이 견디는 선량을 늘린 배수", C.accent4);
  textbox(s, [
    { text: "생존 = 무손상 정확도(96.7 %)의 90 % 이상 유지.  위 축은 r = 10" },
    { text: "−6", options: { superscript: true } },
    { text: " /bit/day 로 환산한 LEO 노출 일수." },
  ], { x: 0.6, y: 6.62, w: 8.3, h: 0.35, fontSize: 10, color: C.accent5 });
  s.addNotes("발표자 2. 핵심 결과.");
}

// ======================= 6. 결과 2: 이봉 분포 =======================
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "결과" });
  s.addText("결과 ② 같은 선량에서도 '멀쩡'과 '붕괴' 두 무리로 갈라진다", { placeholder: "title" });
  s.addImage({ path: path.join(__dirname, "fig_bimodal_a.png"), x: 0.6, y: 1.45, w: 6.6, h: 5.2, objectName: "그림: 정확도 분포 (이봉)" });
  textbox(s, "왜 두 무리일까?", { x: 7.6, y: 1.5, w: 5.1, h: 0.5, fontSize: 20, bold: true });
  bullets(s, [
    "p = 0.03~0.5 %, 50회 시행의 정확도 분포. 중간값은 거의 없고 100 % 근처와 10~50 %에 몰려 있다.",
    "운명을 가르는 것은 '명중 수'가 아니라 bit 30(지수 맨 앞) 명중이 한 번이라도 있었는가.",
    "명중 k번 중 bit 30 을 한 번도 안 맞을 확률 = (31/32)^k",
  ], { x: 7.6, y: 2.1, w: 5.1, h: 2.6, fontSize: 14, color: C.text1 });
  stat(s, 7.6, 4.85, 2.45, "37 %", "k = 31번 명중\n(p = 0.03 %)", C.accent4);
  stat(s, 10.25, 4.85, 2.45, "4 %", "k = 104번 명중\n(p = 0.1 %)", C.accent1);
  textbox(s, "그래서 평균 대신 중앙값과 생존률(90 % 이상 유지한 시행의 비율)로 결과를 읽는다.", { x: 7.6, y: 6.5, w: 5.1, h: 0.5, fontSize: 11, color: C.accent5 });
  s.addNotes("발표자 2. 이봉 분포와 bit 30.");
}

// ======================= 7. 결과 3: 비트별 치명도 =======================
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "결과" });
  s.addText("결과 ③ 32비트 가운데 치명적인 자리는 단 하나, bit 30", { placeholder: "title" });
  s.addImage({ path: FIG + "fig_e2_bits.png", x: 0.6, y: 1.4, w: 12.1, h: 3.68, objectName: "그림: 비트별 단일 명중 생존률" });
  const facts = [
    ["bit 30 한 번", "생존률 0.74 (mnist_linear)\n네 모델 모두 0.5~0.8", C.accent1],
    ["나머지 31개 비트", "생존률 1.0\n부호·가수·낮은 지수 비트는 무해", C.accent5],
    ["모형의 예측", "1 − 32q = 0.72\n실측 0.74 와 일치", C.accent4],
  ];
  facts.forEach(([h, b, col], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 5.3, 3.85, 1.4);
    n++; s.addShape(pres.ShapeType.rect, { x: x + 0.25, y: 5.55, w: 0.18, h: 0.9, fill: { color: col }, line: { color: col }, objectName: "bar" + i });
    textbox(s, h, { x: x + 0.65, y: 5.5, w: 3.0, h: 0.4, fontSize: 15, bold: true, valign: "middle" });
    textbox(s, b, { x: x + 0.65, y: 5.9, w: 3.05, h: 0.75, fontSize: 12, color: C.text2, valign: "top" });
  });
  s.addNotes("발표자 2. 비트 지도.");
}

// ======================= 8. 결과 4: 모델 크기 =======================
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "결과" });
  s.addText("결과 ④ 생존은 '명중 수'가 정한다. 큰 모델이 먼저 죽는다", { placeholder: "title" });
  const hdr = { bold: true, color: C.background1, fill: { color: THEME.colors.dk1 }, align: "center", valign: "middle", fontSize: 13 };
  const cell = (t, o = {}) => ({ text: t, options: Object.assign({ fontSize: 13, align: "center", valign: "middle", color: C.text1 }, o) });
  const rows = [
    [cell("모델", hdr), cell("가중치 수", hdr), cell("하루 명중 수", hdr), cell("생존 50 % 명중 수 (k50)", hdr), cell("버티는 시간 (방어 없음)", hdr), cell("버티는 시간 (클리핑)", hdr)],
    [cell("mnist_linear (FC 2층)", { align: "left" }), cell("101,632"), cell("3.3"), cell("56"), cell("17일", { bold: true, color: C.accent1 }), cell("1,754일", { bold: true, color: C.accent2 })],
    [cell("mnist_cnn (Conv 2 + FC 2)", { align: "left" }), cell("220,064"), cell("7.0"), cell("66"), cell("9일", { bold: true, color: C.accent1 }), cell("937일", { bold: true, color: C.accent2 })],
    [cell("cifar_deep (Conv 6 + FC 1)", { align: "left" }), cell("307,040"), cell("9.8"), cell("64"), cell("7일", { bold: true, color: C.accent1 }), cell("94일", { bold: true, color: C.accent2 })],
    [cell("resnet18_t10 (ResNet-18)", { align: "left" }), cell("11,172,032"), cell("357.5"), cell("112"), cell("7.5시간", { bold: true, color: C.accent1 }), cell("3일", { bold: true, color: C.accent2 })],
  ];
  s.addTable(rows, { x: 0.6, y: 1.5, w: 12.1, colW: [3.0, 1.6, 1.6, 2.2, 1.85, 1.85], rowH: 0.52, border: { type: "solid", color: THEME.colors.lt2, pt: 1 }, fill: { color: THEME.colors.lt1 }, objectName: "표: 모델 4종 비교" });
  stat(s, 0.6, 4.75, 3.85, "×110", "가중치 수 차이\n(10만 → 1,100만 개)", C.accent5);
  stat(s, 4.72, 4.75, 3.85, "60 ~ 110", "생존 50 %가 되는 명중 수 — 거의 일정", C.accent4);
  stat(s, 8.85, 4.75, 3.85, "17일 → 7.5시간", "버티는 시간은 모델 크기에 반비례", C.accent1, 30);
  textbox(s, [
    { text: "하루 명중 수 = r × 32 × 가중치 수 (r = 10" },
    { text: "−6", options: { superscript: true } },
    { text: " /bit/day). 가중치가 110배 많으면 하루에 110배 더 맞는다." },
  ], { x: 0.6, y: 6.45, w: 12.1, h: 0.4, fontSize: 11, color: C.accent5 });
  s.addNotes("발표자 2. 모델 크기 축.");
}

// ======================= 9. 결과 5: 방어 비교 =======================
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "결과" });
  s.addText("결과 ⑤ 방어 6종 비교: 비트를 지키는 방어가 이긴다", { placeholder: "title" });
  s.addImage({ path: FIG + "fig_e5_defenses.png", x: 0.6, y: 1.45, w: 7.9, h: 3.25, objectName: "그림: 방어 6종 비교" });
  const defs = [
    ["패리티", "약 510배", "비트 오류 검출 → 가중치 0으로. 20 % 지워져도 무손상. 메모리 +3 %", C.accent4],
    ["TMR (3중 복제)", "약 170배", "세 복사본의 중앙값. p ≈ 9 % 부근에서 한계. 메모리 +200 %", C.accent3],
    ["클리핑", "약 100배", "NaN·∞ 제거 + [−3,3] 절단. 비용 거의 0", C.accent2],
    ["앙상블 중앙값 / 평균", "2배 / 0.2배", "모델 5개 투표. 평균은 ∞ 하나에 전체가 끌려감. 메모리 +400 %", C.accent1],
  ];
  defs.forEach(([h, g, b, col], i) => {
    const y = 1.45 + i * 1.32;
    card(s, 8.8, y, 3.9, 1.2);
    textbox(s, h, { x: 9.0, y: y + 0.1, w: 2.2, h: 0.4, fontSize: 14, bold: true, valign: "middle" });
    textbox(s, g, { x: 11.2, y: y + 0.1, w: 1.35, h: 0.4, fontSize: 14, bold: true, color: col, align: "right", valign: "middle" });
    textbox(s, b, { x: 9.0, y: y + 0.52, w: 3.55, h: 0.65, fontSize: 11, color: C.text2, valign: "top" });
  });
  textbox(s, "이득 = 생존률 50 %에 이르는 선량이 '방어 없음'의 몇 배인가", { x: 0.6, y: 4.75, w: 7.9, h: 0.35, fontSize: 11, color: C.accent5 });
  card(s, 0.6, 5.2, 7.9, 1.5, { fill: { color: "FFF4E8" }, line: { color: "FFF4E8" } });
  textbox(s, [
    { text: "왜 평균 앙상블은 더 나쁠까? ", options: { bold: true, color: C.accent1 } },
    { text: "다섯 모델 중 하나라도 ∞ 를 내면 평균도 ∞ 가 된다. 다섯 개가 모두 맞아야 하니 혼자일 때보다 5배 자주 죽는다. 중앙값은 3개가 동시에 망가져야 틀리므로 2배 이득을 본다." },
  ], { x: 0.85, y: 5.3, w: 7.4, h: 1.3, fontSize: 13, valign: "middle" });
  s.addNotes("발표자 2. 방어 비교. 끝나면 발표자 1에게 넘긴다.");
}

// ======================= 10. 수학 모형 =======================
pres.addSection({ title: "모형과 결론" });
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "모형과 결론" });
  s.addText("한 줄 공식으로 요약: 명중 하나가 치명적일 확률 q ≈ 1/120", { placeholder: "title" });
  card(s, 0.6, 1.4, 12.1, 1.3, { fill: { color: THEME.colors.dk1 }, line: { color: THEME.colors.dk1 } });
  textbox(s, "생존률  S = e^(−q·k),     k = r × 32 × N_w × T   (명중 수)", { x: 0.9, y: 1.45, w: 11.5, h: 0.7, fontSize: 26, bold: true, color: C.background1, align: "center", valign: "middle" });
  textbox(s, "k번 맞았는데 한 번도 치명적이지 않을 확률. 주사위를 k번 던져 한 번도 1이 안 나올 확률과 같은 꼴이다.", { x: 0.9, y: 2.15, w: 11.5, h: 0.45, fontSize: 14, color: C.accent3, align: "center" });
  textbox(s, [
    { text: "q 는 모델의 구조에서 거의 계산된다:   q ≈ (1/32) × π" },
    { text: "+", options: { subscript: true } },
    { text: " × α" },
  ], { x: 0.6, y: 2.9, w: 12.1, h: 0.45, fontSize: 18, bold: true });
  const fac = [
    ["1/32", "자리", "32비트 중 bit 30 이 뽑힐 확률", C.accent1],
    [[{ text: "π" }, { text: "+", options: { subscript: true } }, { text: " ≈ 0.49" }], "부호", "양수 가중치 비율. 음수는 −∞ 가 되어 ReLU 에서 0으로 잘린다.", C.accent2],
    ["α ≈ 0.56", "경로", "∞ 가 출력까지 실제로 전파되는 비율", C.accent4],
  ];
  fac.forEach(([v, h, b, col], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 3.45, 3.85, 1.6);
    textbox(s, v, { x: x + 0.25, y: 3.55, w: 3.35, h: 0.55, fontSize: 24, bold: true, color: col, valign: "middle" });
    textbox(s, h, { x: x + 0.25, y: 4.1, w: 3.35, h: 0.3, fontSize: 13, bold: true });
    textbox(s, b, { x: x + 0.25, y: 4.4, w: 3.35, h: 0.6, fontSize: 12, color: C.text2, valign: "top" });
  });
  stat(s, 0.6, 5.25, 3.85, "0.0086", "mnist_linear 의 q\n≈ 120번 중 1번", C.accent1);
  stat(s, 4.72, 5.25, 3.85, "오차 0.06", "맞추지 않은 궤도 모드(SAA) 데이터 38점의\n생존률을 이 공식이 예측한 평균 오차", C.accent4);
  stat(s, 8.85, 5.25, 3.85, [{ text: "k" }, { text: "50", options: { subscript: true } }, { text: " ≈ 80" }], "ln2 / q — 네 모델 실측 60~110 과 같은 수준", C.accent2);
  s.addNotes("발표자 1. 수학 모형.");
}
// stat 카드 높이 1.55 → y 5.85 + 1.55 = 7.4 > footer. 위 슬라이드에서는 카드가 footer 와 겹치므로 아래에서 보정한다.

// ======================= 11. 결론 =======================
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: "모형과 결론" });
  s.addText("결론: 위성 속 AI에 방어는 선택이 아니라 필수", { placeholder: "title", fontSize: 32 });
  const pts = [
    ["1", "정확도는 매끄럽게 줄지 않는다", "bit 30 명중 하나로 '멀쩡'과 '붕괴'가 갈린다."],
    ["2", "버티는 시간은 모델 크기에 반비례", "ResNet-18 급은 방어 없이 하루를 못 넘긴다."],
    ["3", "클리핑만 켜도 100배, 지수 비트 보호가 최선", "비용이 거의 없는 클리핑을 기본으로, bit 30 을 패리티·TMR 로 선택 보호."],
  ];
  pts.forEach(([num, h, b], i) => {
    const y = 4.0 + i * 0.95;
    numCircle(s, 0.8, y + 0.05, num, C.accent3);
    textbox(s, h, { x: 1.5, y, w: 11, h: 0.4, fontSize: 18, bold: true, color: C.background1, valign: "middle" });
    textbox(s, b, { x: 1.5, y: y + 0.4, w: 11, h: 0.4, fontSize: 13, color: C.accent5, valign: "top" });
  });
  textbox(s, "감사합니다.   코드·데이터: github.com/SJinLee/AI-in-Space", { x: 0.8, y: 6.85, w: 11.7, h: 0.4, fontSize: 13, color: C.accent3 });
  s.addNotes("발표자 1·2. 결론과 마무리.");
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
