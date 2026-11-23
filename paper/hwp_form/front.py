"""대회 양식(hwpx)의 제목·저자·초록 칸을 채우고, 안내문·메모·본문 예시를 지운다.

사용법: python front.py <양식.hwpx> <abstract.txt> <front.hwpx>
"""
import re, sys, zipfile
from xml.sax.saxutils import escape

tpl, absf, out = sys.argv[1:4]

TITLE = ['저궤도 우주 방사선 단일 사건 오류(SEU)가', '인공신경망 가중치에 미치는 영향']
SUBTITLE = '— 모델 크기·선량·방어에 따른 생존률의 측정과 예측 모형 —'
STUDENTS = ['이태린 (민락중학교)', '최재원 (선덕중학교)']
STAFF = ['지도교수 : 이성진 (대진대학교)', '조    교 : 이유림 (대진대학교)']

z = zipfile.ZipFile(tpl)
infos = z.infolist()
files = {i.filename: z.read(i.filename) for i in infos}
s = files['Contents/section0.xml'].decode('utf-8')
s = re.sub(r'<hp:linesegarray>.*?</hp:linesegarray>', '', s, flags=re.S)


def para(ppr, cpr, text=''):
    t = f'<hp:t>{escape(text)}</hp:t>' if text else ''
    return (f'<hp:p id="0" paraPrIDRef="{ppr}" styleIDRef="14" pageBreak="0" columnBreak="0" merged="0">'
            f'<hp:run charPrIDRef="{cpr}">{t}</hp:run></hp:p>')


# 1) 제목(메모 포함)과 그 뒤 저자 줄들을 새로 쓴다.
a = s.index('<hp:ctrl><hp:fieldBegin')
assert 'type="MEMO"' in s[a:a + 200]
b = s.index('<hp:t>논문제목</hp:t>', a)
tbl_p = s.rfind('<hp:p ', 0, s.index('<hp:tbl'))        # 초록 표가 들어 있는 문단
title = '<hp:t>' + '<hp:lineBreak/>'.join(escape(x) for x in TITLE) + '</hp:t></hp:run></hp:p>'
block = [para(2, 12, SUBTITLE), para(14, 9)]
block += [para(14, 9, x) for x in STUDENTS] + [para(14, 9)]
block += [para(14, 9, x) for x in STAFF] + [para(13, 9)]
s = s[:a] + title + ''.join(block) + s[tbl_p:]
assert '홍길동' not in s[:s.index('<hp:tbl')]

# 2) 초록 칸
abstract = open(absf, encoding='utf-8').read().strip()
s, n = re.subn(r'<hp:run charPrIDRef="16"><hp:ctrl><hp:fieldBegin id="\d+" type="CLICK_HERE".*?<hp:fieldEnd[^>]*/></hp:ctrl><hp:t> </hp:t></hp:run>',
               lambda m: '<hp:run charPrIDRef="16"><hp:t>' + escape(abstract) + '</hp:t></hp:run>', s, flags=re.S)
assert n == 1

# 3) 표 뒤의 예시 본문을 모두 지우고 빈 문단 하나만 남긴다(본문을 끼워 넣을 자리).
e = s.index('</hp:tbl>')
e = s.index('</hp:p>', e) + len('</hp:p>')
s = s[:e] + para(13, 15) + '</hs:sec>'
assert '↳' not in s and '서론' not in s

files['Contents/section0.xml'] = s.encode('utf-8')
with zipfile.ZipFile(out, 'w') as zo:
    for i in infos:
        zo.writestr(i, files[i.filename], compress_type=i.compress_type)
print('front ok', len(s))
