// A1 세로 포스터 — pptxgenjs
const pptxgen = require("pptxgenjs");
const path = require("path");
const { applyTheme } = require("C:/Users/math/.claude/skills/synced/c3517a04-a7ed-4c5f-aafd-36f5f56c8606_019c1aca-84c5-42ca-9ebf-35917cb803a7/pptx/scripts/apply_theme.js");

const FIG = "C:/Users/math/Dropbox/daejin(me)/영재/2026/AI-in-space/paper/figures/";
const OUT = process.argv[2];

const THEME = {
  name: "SEU Space Poster",
  headFontFace: "Malgun Gothic",
  bodyFontFace: "Malgun Gothic",
  colors: {
    dk1: "0B1B3A", lt1: "FFFFFF", dk2: "4A566E", lt2: "EEF2F8",
    accent1: "D9532B", accent2: "2E6DB4", accent3: "F2B134",
    accent4: "2A9D8F", accent5: "8A94A6", accent6: "6C3FB5",
    hlink: "2E6DB4", folHlink: "6C3FB5",
  },
};

const W = 23.39, H = 33.11; // A1 portrait (inches)
const pres = new pptxgen();
pres.defineLayout({ name: "A1", width: W, height: H });
pres.layout = "A1";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "우주에서 AI는 살아남을까? (포스터)";
pres.author = "이태린, 최재원";
const C = pres.SchemeColor;

pres.defineSlideMaster({ title: "POSTER", background: { color: THEME.colors.lt1 }, objects: [] });
const s = pres.addSlide({ masterName: "POSTER" });

let n = 0;
const T = (text, o) => { n++; return s.addText(text, Object.assign({ isTextBox: true, margin: 0, fontSize: 18, color: C.text1, objectName: "t" + n }, o)); };
const card = (x, y, w, h, o = {}) => { n++; s.addShape(pres.ShapeType.roundRect, Object.assign({ x, y, w, h, rectRadius: 0.2, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "c" + n }, o)); };
const rect = (x, y, w, h, col, o = {}) => { n++; s.addShape(pres.ShapeType.rect, Object.assign({ x, y, w, h, fill: { color: col }, line: { color: col }, objectName: "r" + n }, o)); };
const bullets = (items, o) => T(items.map((t, i) => ({ text: t, options: { bullet: { indent: 24 }, breakLine: i < items.length - 1, paraSpaceAfter: 10 } })), Object.assign({ fontSize: 19, valign: "top" }, o));
function section(x, y, w, num, title, col) {
  n++; s.addShape(pres.ShapeType.ellipse, { x, y, w: 0.75, h: 0.75, fill: { color: col }, line: { color: col }, objectName: "sn" + n });
  T(String(num), { x, y, w: 0.75, h: 0.75, fontSize: 26, bold: true, color: C.background1, align: "center", valign: "middle" });
  T(title, { x: x + 0.95, y, w: w - 0.95, h: 0.75, fontSize: 30, bold: true, valign: "middle" });
}
function stat(x, y, w, h, big, label, col, bigSize = 46) {
  card(x, y, w, h);
  T(big, { x: x + 0.2, y: y + 0.1, w: w - 0.4, h: h * 0.55, fontSize: bigSize, bold: true, color: col, align: "center", valign: "middle" });
  T(label, { x: x + 0.2, y: y + h * 0.62, w: w - 0.4, h: h * 0.35, fontSize: 15, color: C.text2, align: "center", valign: "top" });
}

// ---------------- header ----------------
rect(0, 0, W, 4.6, THEME.colors.dk1);
// stars
let seed = 11; const rnd = () => { seed = (seed * 9301 + 49297) % 233280; return seed / 233280; };
for (let i = 0; i < 90; i++) { const d = 0.05 + rnd() * 0.1; n++; s.addShape(pres.ShapeType.ellipse, { x: rnd() * (W - 0.3), y: rnd() * 4.3, w: d, h: d, fill: { color: C.background1, transparency: Math.round(rnd() * 55) }, line: { type: "none" }, objectName: "star" + i }); }
T("2026 영재교육 · AI in Space", { x: 1.0, y: 0.5, w: 12, h: 0.5, fontSize: 20, bold: true, color: C.accent3 });
T("우주에서 AI는 살아남을까?", { x: 1.0, y: 1.0, w: 17, h: 1.5, fontSize: 64, bold: true, color: C.background1, valign: "middle" });
T("저궤도 우주 방사선이 뒤집은 비트 하나가 인공신경망 가중치에 미치는 영향 — 생존률의 측정과 예측 모형", { x: 1.0, y: 2.5, w: 17.5, h: 0.7, fontSize: 24, color: C.background1, valign: "middle" });
T([
  { text: "이태린", options: { bold: true } }, { text: " (민락중학교)   " },
  { text: "최재원", options: { bold: true } }, { text: " (선덕중학교)      지도교사 이성진 · 조교 이유림 (대진대학교)" },
], { x: 1.0, y: 3.35, w: 17.5, h: 0.6, fontSize: 21, color: C.accent5, valign: "middle" });
// orbit graphic
n++; s.addShape(pres.ShapeType.ellipse, { x: 19.7, y: 1.1, w: 2.6, h: 2.6, fill: { color: C.accent2 }, line: { color: C.accent2 }, objectName: "earth" });
n++; s.addShape(pres.ShapeType.ellipse, { x: 18.9, y: 0.55, w: 4.2, h: 3.7, fill: { type: "none" }, line: { color: C.accent3, width: 2.5, dashType: "dash" }, objectName: "orbit" });
rect(22.75, 2.1, 0.3, 0.3, THEME.colors.accent3);

const LX = 0.9, LW = 10.7, RX = 12.2, RW = 10.3;

// ---------------- LEFT 1: 배경과 질문 ----------------
let y = 5.2;
section(LX, y, LW, 1, "배경과 연구 질문", C.accent1);
bullets([
  "우주의 고에너지 입자가 메모리 셀의 0↔1 을 바꾸는 단일 사건 오류(SEU). 저궤도 상용 메모리 1 GB 는 하루 약 8,600번(r = 10^-6 /bit/day), 남대서양 이상 지역(SAA) 통과 때 집중.",
  "위성 영상 판독·자율 항법용 신경망을 위성에 직접 싣는 시도가 늘고 있다. 신경망의 가중치도 메모리에 있다.",
], { x: LX, y: y + 1.0, w: LW, h: 2.6, fontSize: 19 });
card(LX, y + 3.6, LW, 1.25, { fill: { color: "FFF4E8" }, line: { color: "FFF4E8" } });
T([{ text: "질문  ", options: { bold: true, color: C.accent1 } }, { text: "가중치의 비트 하나가 뒤집히면 무슨 일이 생길까? 얼마나 버틸까? 어떻게 막을까?", options: { bold: true } }],
  { x: LX + 0.3, y: y + 3.6, w: LW - 0.6, h: 1.25, fontSize: 20, valign: "middle" });

// ---------------- LEFT 2: float32 ----------------
y = 10.5;
section(LX, y, LW, 2, "가중치 하나 = float32 32비트", C.accent2);
{
  const by = y + 1.05, bh = 0.8, unit = LW / 32; let cx = LX;
  [[1, C.accent2, "", "bit 31 부호"], [8, C.accent1, "지수 (8비트)", "bit 30–23"], [23, C.accent5, "가수 (23비트)", "bit 22–0"]].forEach(([b, col, lab, rng]) => {
    rect(cx, by, unit * b, bh, col, { line: { color: C.background1, width: 1.5 } });
    if (lab) T(lab, { x: cx, y: by, w: unit * b, h: bh, fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle" });
    T(rng, { x: cx, y: by + bh + 0.05, w: Math.max(unit * b, 2.2), h: 0.35, fontSize: 14, color: C.text2, align: b === 1 ? "left" : "center" });
    cx += unit * b;
  });
  T("값 = (−1)^부호 × 2^(지수−127) × 1.가수",  { x: LX, y: by + 1.3, w: LW, h: 0.4, fontSize: 17, color: C.text2 });
  const ex = [["bit 0", "1.0 → 1.0000001", "거의 변화 없음", C.accent5], ["bit 22", "1.0 → 1.5", "조금 커짐 · 무해", C.accent5], ["bit 31 (부호)", "1.0 → −1.0", "부호만 반전", C.accent2], ["bit 30 (지수 맨 앞)", "1.0 → +∞", "×2^128 · 치명적", C.accent1]];
  ex.forEach(([a, b, c, col], i) => {
    const x = LX + i * (LW / 4), w = LW / 4 - 0.2, yy = by + 1.85;
    card(x, yy, w, 1.9);
    rect(x + 0.25, yy + 0.25, 0.35, 0.35, col);
    T(a, { x: x + 0.7, y: yy + 0.2, w: w - 0.8, h: 0.45, fontSize: 15, bold: true, valign: "middle" });
    T(b, { x: x + 0.25, y: yy + 0.75, w: w - 0.4, h: 0.5, fontSize: 16, bold: true, color: col, valign: "middle" });
    T(c, { x: x + 0.25, y: yy + 1.3, w: w - 0.4, h: 0.45, fontSize: 14, color: C.text2 });
  });
  T("bit 30 이 0→1 이 되면 값이 2^128 ≈ 3×10^38 배. 한 층만 지나도 출력 전체를 지배한다.", { x: LX, y: by + 3.95, w: LW, h: 0.5, fontSize: 18, bold: true, color: C.accent1 });
}

// ---------------- LEFT 3: 방법 ----------------
y = 16.6;
section(LX, y, LW, 3, "실험 방법: 비트를 진짜로 XOR 하는 시뮬레이터", C.accent4);
{
  const steps = [["학습 끝난 모델", "MNIST 손글씨 분류, 가중치 101,632개 = 3,252,224비트", C.accent2], ["가중치 뽑기", "각 가중치를 확률 p 로 선택. p = 0.001 % ~ 10 %, 13단계", C.accent4], ["비트 하나 XOR", "32비트 중 무작위 한 자리를 0↔1 (가우시안 잡음 아님)", C.accent1], ["정확도 측정", "시험 10,000장 채점. p 마다 10회, 총 260회", C.accent3]];
  steps.forEach(([h, b, col], i) => {
    const x = LX + (i % 2) * (LW / 2), yy = y + 1.05 + Math.floor(i / 2) * 1.55, w = LW / 2 - 0.2;
    card(x, yy, w, 1.4);
    n++; s.addShape(pres.ShapeType.ellipse, { x: x + 0.25, y: yy + 0.25, w: 0.55, h: 0.55, fill: { color: col }, line: { color: col }, objectName: "st" + i });
    T(String(i + 1), { x: x + 0.25, y: yy + 0.25, w: 0.55, h: 0.55, fontSize: 17, bold: true, color: C.background1, align: "center", valign: "middle" });
    T(h, { x: x + 0.95, y: yy + 0.2, w: w - 1.1, h: 0.6, fontSize: 19, bold: true, valign: "middle" });
    T(b, { x: x + 0.25, y: yy + 0.8, w: w - 0.45, h: 0.55, fontSize: 14, color: C.text2 });
  });
  T([
    { text: "방어 없음 (none)", options: { bold: true, color: C.accent1 } }, { text: " 뒤집힌 그대로 사용   " },
    { text: "클리핑 (clip)", options: { bold: true, color: C.accent2 } }, { text: " NaN·∞ → 0, 값을 [−3, 3] 으로 절단", options: { breakLine: true } },
    { text: "확장: ", options: { bold: true } }, { text: "모델 4종(가중치 10만~1,100만 개, ResNet-18 포함) · 비트 한 자리만 뒤집기(E2) · SAA 궤도 모드(E3) · 방어 6종(E5)" },
  ], { x: LX, y: y + 4.25, w: LW, h: 1.2, fontSize: 16, valign: "top" });
}

// ---------------- LEFT 4: 수학 모형 ----------------
y = 22.4;
section(LX, y, LW, 4, "한 줄 공식: 명중 하나가 치명적일 확률 q", C.accent6);
card(LX, y + 1.05, LW, 1.7, { fill: { color: THEME.colors.dk1 }, line: { color: THEME.colors.dk1 } });
T("S = e^(−q·k),   k = r × 32 × N_w × T", { x: LX + 0.3, y: y + 1.1, w: LW - 0.6, h: 1.0, fontSize: 34, bold: true, color: C.background1, align: "center", valign: "middle" });
T("S 생존률 · k 명중 수 · r 발생률 · N_w 가중치 수 · T 노출 일수", { x: LX + 0.3, y: y + 2.1, w: LW - 0.6, h: 0.5, fontSize: 15, color: C.accent3, align: "center" });
{
  const fac = [["1/32", "bit 30 이 뽑힐 확률", C.accent1], ["π+ ≈ 0.49", "양수 가중치 비율", C.accent2], ["α ≈ 0.56", "∞ 가 출력까지 전파되는 비율", C.accent4]];
  T("q ≈ (1/32) × π+ × α ≈ 0.0086  (120번에 1번)", { x: LX, y: y + 2.9, w: LW, h: 0.5, fontSize: 21, bold: true });
  fac.forEach(([v, b, col], i) => {
    const x = LX + i * (LW / 3), w = LW / 3 - 0.2;
    card(x, y + 3.5, w, 1.35);
    T(v, { x: x + 0.25, y: y + 3.55, w: w - 0.4, h: 0.7, fontSize: 26, bold: true, color: col, valign: "middle" });
    T(b, { x: x + 0.25, y: y + 4.25, w: w - 0.4, h: 0.5, fontSize: 14, color: C.text2 });
  });
  T("stress 조건으로 맞춘 공식이, 맞추지 않은 SAA 궤도 모드 38점의 생존률을 평균 오차 0.06 으로 예측. 단일 bit 30 명중 생존률 1−32q = 0.72 도 실측 0.74 와 일치.", { x: LX, y: y + 5.0, w: LW, h: 0.9, fontSize: 16, color: C.text2 });
}

// ---------------- LEFT 5: 결론 ----------------
y = 28.6;
card(LX, y, LW, 3.75, { fill: { color: THEME.colors.dk1 }, line: { color: THEME.colors.dk1 } });
T("결론: 위성 속 AI에 방어는 선택이 아니라 필수", { x: LX + 0.4, y: y + 0.2, w: LW - 0.8, h: 0.7, fontSize: 28, bold: true, color: C.accent3, valign: "middle" });
T([
  { text: "① 정확도는 매끄럽게 줄지 않는다. ", options: { bold: true } }, { text: "bit 30 명중 하나로 '멀쩡'과 '붕괴'가 갈린다.", options: { breakLine: true } },
  { text: "② 버티는 시간은 모델 크기에 반비례. ", options: { bold: true } }, { text: "ResNet-18 급은 방어 없이 하루를 못 넘긴다.", options: { breakLine: true } },
  { text: "③ 클리핑만 켜도 10~100배. ", options: { bold: true } }, { text: "MNIST 두 모델 약 100배, CIFAR·ResNet 10배대. 비용 0 인 클리핑을 기본으로 하고, 치명적인 bit 30 만 패리티·TMR 로 골라 보호하는 방법을 다음 연구로 제안한다." },
], { x: LX + 0.4, y: y + 1.0, w: LW - 0.8, h: 2.6, fontSize: 18, color: C.background1, valign: "top", paraSpaceAfter: 8 });

// ================= RIGHT 1: 결과 ① =================
y = 5.2;
section(RX, y, RW, 5, "결과 ① 방어 없음 17일, 클리핑 1,754일", C.accent1);
s.addImage({ path: FIG + "fig_e1_dose_response.png", x: RX, y: y + 1.0, w: RW, h: RW * 800 / 1280, objectName: "그림 1" });
{
  const yy = y + 1.0 + RW * 800 / 1280 + 0.2, w = RW / 3 - 0.2;
  stat(RX, yy, w, 1.9, "17일", "방어 없음 · 생존률 50 % (p ≈ 0.055 %)", C.accent1);
  stat(RX + RW / 3, yy, w, 1.9, "1,754일", "클리핑 · 생존률 50 % (p ≈ 5.6 %)", C.accent2);
  stat(RX + 2 * RW / 3, yy, w, 1.9, "×102", "클리핑이 견디는 선량을 늘린 배수", C.accent4);
  T("생존 = 무손상 정확도(96.7 %)의 90 % 이상 유지. 위 축은 r = 10^-6 /bit/day 로 환산한 LEO 노출 일수.", { x: RX, y: yy + 2.0, w: RW, h: 0.4, fontSize: 13, color: C.accent5 });
}

// ================= RIGHT 2: 결과 ② ③ =================
y = 16.0;
section(RX, y, RW, 6, "결과 ②③ 두 무리로 갈라진다 — 범인은 bit 30", C.accent2);
s.addImage({ path: FIG + "fig_bimodal_a.png", x: RX, y: y + 1.0, w: 5.0, h: 5.0 * 660 / 767, objectName: "그림 2" });
bullets([
  "p = 0.03~0.5 %, 500회 시행(5단계 × 100회)의 정확도 분포. 중간값이 거의 없고 100 % 근처와 10~50 % 에 몰린다 (왼쪽).",
  "명중 k번 중 bit 30 을 한 번도 안 맞을 확률 = (31/32)^k : k=31 → 37 %, k=104 → 4 %.",
  "한 비트만 정확히 뒤집는 실험(아래): 32비트 중 bit 30 만 치명적(생존률 0.53~0.83), 나머지 31개는 1.0.",
], { x: RX + 5.3, y: y + 1.0, w: RW - 5.3, h: 4.4, fontSize: 16 });
s.addImage({ path: FIG + "fig_e2_bits.png", x: RX, y: y + 5.5, w: RW, h: RW * 620 / 2040, objectName: "그림 3" });

// ================= RIGHT 3: 결과 ④ =================
y = 25.0;
section(RX, y, RW, 7, "결과 ④ 생존은 '명중 수'가 정한다", C.accent4);
{
  const hdr = { bold: true, color: C.background1, fill: { color: THEME.colors.dk1 }, align: "center", valign: "middle", fontSize: 14 };
  const cell = (t, o = {}) => ({ text: t, options: Object.assign({ fontSize: 14, align: "center", valign: "middle", color: C.text1 }, o) });
  const rows = [
    [cell("모델", hdr), cell("가중치 수", hdr), cell("하루 명중", hdr), cell("k50", hdr), cell("버팀 (없음)", hdr), cell("버팀 (클리핑)", hdr)],
    [cell("mnist_linear", { align: "left" }), cell("101,632"), cell("3.3"), cell("56"), cell("17일", { bold: true, color: C.accent1 }), cell("1,754일", { bold: true, color: C.accent2 })],
    [cell("mnist_cnn", { align: "left" }), cell("220,064"), cell("7.0"), cell("66"), cell("9일", { bold: true, color: C.accent1 }), cell("937일", { bold: true, color: C.accent2 })],
    [cell("cifar_deep", { align: "left" }), cell("307,040"), cell("9.8"), cell("64"), cell("7일", { bold: true, color: C.accent1 }), cell("94일", { bold: true, color: C.accent2 })],
    [cell("resnet18_t10", { align: "left" }), cell("11,172,032"), cell("357.5"), cell("112"), cell("7.5시간", { bold: true, color: C.accent1 }), cell("3일", { bold: true, color: C.accent2 })],
  ];
  s.addTable(rows, { x: RX, y: y + 1.0, w: RW, colW: [2.3, 1.7, 1.4, 1.1, 1.9, 1.9], rowH: 0.42, border: { type: "solid", color: THEME.colors.lt2, pt: 1 }, fill: { color: THEME.colors.lt1 }, objectName: "표 1" });
  T("가중치 수가 110배 차이 나도 생존률 50 % 명중 수 k50 은 56~112 로 거의 일정. 하루 명중 수 = r × 32 × N_w 이므로 버티는 시간 ∝ 1 / N_w.", { x: RX, y: y + 3.2, w: RW, h: 0.8, fontSize: 15, color: C.text2 });
}

// ================= RIGHT 4: 결과 ⑤ =================
y = 29.2;
section(RX, y, RW, 8, "결과 ⑤ 방어 6종: 비트를 지키는 방어가 이긴다", C.accent3);
{
  const defs = [["패리티", "≈510배", "+3 % 메모리", C.accent4], ["TMR", "≈170배", "+200 %", C.accent3], ["클리핑", "≈100배", "비용 0 (MNIST)", C.accent2], ["앙상블 중앙값", "2배", "+400 %", C.accent5], ["앙상블 평균", "0.2배", "해로움", C.accent1]];
  defs.forEach(([h, g, b, col], i) => {
    const w = RW / 5 - 0.15, x = RX + i * (RW / 5), yy = y + 1.0;
    card(x, yy, w, 1.9);
    T(h, { x: x + 0.15, y: yy + 0.1, w: w - 0.3, h: 0.45, fontSize: 15, bold: true, align: "center", valign: "middle" });
    T(g, { x: x + 0.15, y: yy + 0.55, w: w - 0.3, h: 0.75, fontSize: 26, bold: true, color: col, align: "center", valign: "middle" });
    T(b, { x: x + 0.15, y: yy + 1.35, w: w - 0.3, h: 0.4, fontSize: 13, color: C.text2, align: "center" });
  });
  T("이득 = 생존률 50 % 선량이 방어 없음의 몇 배인가. 평균 앙상블은 다섯 모델 중 하나만 ∞ 를 내도 평균이 ∞ 가 되어 혼자일 때의 1/5 선량에서 죽는다(실측 이득 0.2배).", { x: RX, y: y + 3.0, w: RW, h: 0.75, fontSize: 14, color: C.text2 });
}

// footer
T("코드·데이터·논문: github.com/SJinLee/AI-in-Space      시뮬레이터는 파이썬으로 직접 구현, 논문의 그림·수치는 CSV 에서 자동 생성", { x: 0.9, y: 32.5, w: 10.7, h: 0.4, fontSize: 13, color: C.accent5 });

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
