set -e
S="$(cd "$(dirname "$0")" && pwd)"; cd "$S"
python prep.py "c:/Users/math/Dropbox/daejin(me)/영재/2026/AI-in-space/paper" . | head -1
"$LOCALAPPDATA/Pandoc/pandoc.exe" main_hwp.tex -f latex -t docx -o main0.docx --reference-doc=ref.docx --lua-filter=tables.lua -M subtitle="— 모델 크기·선량·방어에 따른 생존률의 측정과 예측 모형 —" -M abstract-title="초록"
python fix_docx.py main0.docx main.docx
W=$(cygpath -w "$S")
timeout 580 python build_hwp.py "$W\main.docx" "$W\test.hwp" "$W\test.pdf" maths.json
python post.py test.hwp final.hwp final.pdf
rm -rf pg3; mkdir pg3; pdftoppm -r 60 -png final.pdf pg3/c
n=$(ls pg3 | wc -l); echo pages $n
for k in 0 6 12 18 24 30; do [ $k -lt $n ] && python sheet.py pg3 s_$k.png $k; done; true
