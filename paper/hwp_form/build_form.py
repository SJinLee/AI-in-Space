"""양식 앞부분(front.hwpx) 뒤에 본문(main.docx)을 끼워 넣고, 수식 자리표시를 한글 수식으로 바꾼다.

사용법: python build_form.py <main.docx> <front.hwpx> <body.hwp> <out.hwp> <out.pdf> <maths.json>
"""
import sys, os, json, time
import win32com.client as win32

docx, front, bodyh, out, pdf, mj = [os.path.abspath(a) for a in sys.argv[1:7]]
maths = json.load(open(mj, encoding="utf-8"))
hwp = win32.gencache.EnsureDispatch("HWPFrame.HwpObject")
hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModule")
hwp.SetMessageBoxMode(0x00010000 | 0x00000001 | 0x00000010)
missing = []
try:
    # 1) docx 본문을 hwp 로 한 번 저장한다.
    print("open docx", hwp.Open(docx, "", ""), hwp.PageCount)
    print("save body", hwp.SaveAs(bodyh, "HWP", ""))
    hwp.Clear(1)
    # 2) 양식 앞부분을 열고 문서 끝에 본문을 끼워 넣는다(쪽 설정은 양식 것을 따른다).
    print("open front", hwp.Open(front, "HWPX", ""), hwp.PageCount)
    hwp.HAction.Run("MoveDocEnd")
    ins = hwp.HParameterSet.HInsertFile
    hwp.HAction.GetDefault("InsertFile", ins.HSet)
    ins.filename = bodyh
    ins.KeepSection = 0
    ins.KeepCharshape = 1
    ins.KeepParashape = 1
    ins.KeepStyle = 1
    print("insert", hwp.HAction.Execute("InsertFile", ins.HSet), hwp.PageCount)

    tables = []
    c = hwp.HeadCtrl
    while c:
        if c.CtrlID == "tbl":
            tables.append(c)
        c = c.Next
    for c in tables[1:]:
        pr = c.Properties; pr.SetItem("TreatAsChar", 1); c.Properties = pr
    print("tables", len(tables))

    t0 = time.time()
    for i, m in enumerate(maths):
        tok = "MQ%04dQM" % i
        hwp.HAction.Run("MoveDocBegin")
        fs = hwp.HParameterSet.HFindReplace
        hwp.HAction.GetDefault("RepeatFind", fs.HSet)
        fs.FindString = tok
        fs.Direction = hwp.FindDir("AllDoc")
        fs.IgnoreMessage = 1
        fs.WholeWordOnly = 0
        fs.FindRegExp = 0
        if not hwp.HAction.Execute("RepeatFind", fs.HSet):
            missing.append(tok); continue
        h = hwp.CharShape.Item("Height")
        hwp.HAction.Run("Delete")
        eq = hwp.HParameterSet.HEqEdit
        hwp.HAction.GetDefault("EquationCreate", eq.HSet)
        eq.EqFontName = "HYhwpEQ"
        eq.string = m["hwp"]
        eq.BaseUnit = h
        hwp.HAction.Execute("EquationCreate", eq.HSet)
        hwp.FindCtrl()
        hwp.HAction.GetDefault("EquationModify", eq.HSet)
        eq.string = eq.VisualString
        eq.Version = "Equation Version 60"
        hwp.HAction.Execute("EquationModify", eq.HSet)
        c = hwp.CurSelectedCtrl
        if c is not None:
            pr = c.Properties
            pr.SetItem("TreatAsChar", 1)
            c.Properties = pr
        hwp.HAction.Run("Cancel")
        if m["display"]:
            hwp.HAction.Run("ParagraphShapeAlignCenter")
    print("inserted", len(maths) - len(missing), "missing", missing, round(time.time() - t0), "s")

    # 본문 줄 간격 160 % (표 안은 닿지 않는다. 양식의 제목·저자 줄도 원래 160 % 이다.)
    hwp.HAction.Run("MoveDocBegin")
    hwp.HAction.Run("SelectAll")
    act = hwp.CreateAction("ParagraphShape")
    pset = act.CreateSet()
    pset.SetItem("LineSpacingType", 0)
    pset.SetItem("LineSpacing", 160)
    pset.SetItem("FontLineHeight", 0)
    print("linespacing", act.Execute(pset))
    hwp.HAction.Run("Cancel")
    hwp.HAction.Run("MoveDocBegin")
    print("save", hwp.SaveAs(out, "HWP", ""), "pdf", hwp.SaveAs(pdf, "PDF", ""), "pages", hwp.PageCount)
finally:
    hwp.Clear(1); hwp.Quit()
