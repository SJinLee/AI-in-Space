set -e
S="$(cd "$(dirname "$0")" && pwd)"; cd "$S"
PAPER="c:/Users/math/Dropbox/daejin(me)/영재/2026/AI-in-space/paper"
python prep.py "$PAPER" . | head -2
"$LOCALAPPDATA/Pandoc/pandoc.exe" main_hwp.tex -f latex -t docx -o main0.docx --reference-doc=ref.docx --lua-filter=tables.lua
python fix_docx.py main0.docx main.docx
python front.py tpl.hwpx abstract.txt front.hwpx
timeout 580 python build_form.py main.docx front.hwpx body.hwp test.hwp test.pdf maths.json
python post.py test.hwp final.hwp final.pdf 1
rm -rf pg3; mkdir pg3; pdftoppm -r 60 -png final.pdf pg3/c
echo pages $(ls pg3 | wc -l)
