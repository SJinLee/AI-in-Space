import sys, re, zipfile
src, out = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(src)
files = {n: z.read(n) for n in z.namelist()}
d = files['word/document.xml'].decode('utf-8')
TW = 9072
def tbl(m):
    t = m.group(0)
    ws = [int(x) for x in re.findall(r'<w:gridCol w:w="(\d+)"', t)]
    k = TW / sum(ws)
    t = re.sub(r'<w:gridCol w:w="(\d+)"', lambda g: '<w:gridCol w:w="%d"' % round(int(g.group(1)) * k), t)
    t = t.replace('<w:tblW w:type="pct" w:w="5000" />', '<w:tblW w:type="dxa" w:w="%d" />' % TW)
    return t
d, n = re.subn(r'<w:tbl>.*?</w:tbl>', tbl, d, flags=re.S)
# 코드 문단을 한 칸짜리 표에 넣어 상자로 만든다(문단 테두리는 한글에서 아래 문단과 붙는다).
def codebox(m):
    edge = ''.join('<w:%s w:val="single" w:sz="4" w:space="0" w:color="808080" />' % k for k in ('top', 'left', 'bottom', 'right'))
    return ('<w:tbl><w:tblPr><w:tblW w:type="dxa" w:w="%d" /><w:tblBorders>%s</w:tblBorders>'
            '<w:tblCellMar><w:left w:w="113" w:type="dxa" /><w:right w:w="113" w:type="dxa" /></w:tblCellMar></w:tblPr>'
            '<w:tblGrid><w:gridCol w:w="%d" /></w:tblGrid><w:tr><w:tc><w:tcPr><w:tcW w:w="%d" w:type="dxa" /></w:tcPr>%s</w:tc></w:tr></w:tbl>'
            % (TW, edge, TW, TW, m.group(0)))
d, nbox = re.subn(r'<w:p><w:pPr><w:pStyle w:val="SourceCode" /></w:pPr>.*?</w:p>', codebox, d, flags=re.S)
print('code boxes', nbox)
files['word/document.xml'] = d.encode('utf-8')
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zo:
    for name, b in files.items(): zo.writestr(name, b)
print('tables', n)
