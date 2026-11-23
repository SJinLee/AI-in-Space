"""한글 고유 조판으로 전환하고(Word 호환 해제) 표 안 줄 간격, 행 높이, 표 바깥 여백을 맞춘다.

사용법: python post.py <in.hwp> <out.hwp> <check.pdf>
"""
import sys, os
import win32com.client as win32

src, out, pdf = [os.path.abspath(a) for a in sys.argv[1:4]]
SKIP = int(sys.argv[4]) if len(sys.argv) > 4 else 0
ROW_PAD = 300        # 행마다 더하는 높이(HWPUNIT, 약 1 mm). 고유 조판은 행에 위아래 여백을 두지 않는다.
CELL_SPACING = 130   # 표 안 줄 간격(%)

hwp = win32.gencache.EnsureDispatch("HWPFrame.HwpObject")
hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModule")
hwp.SetMessageBoxMode(0x00010000 | 0x1 | 0x10)


def enter(c):
    """표 c 의 첫 칸에 커서를 둔다. 실패하면 False."""
    for how in ("SelectCtrlFront", "SelectCtrlReverse"):
        hwp.SetPosBySet(c.GetAnchorPos(0))
        hwp.HAction.Run(how)
        sel = hwp.CurSelectedCtrl
        if sel is not None and sel.CtrlID == "tbl":
            hwp.HAction.Run("ShapeObjTableSelCell")
            return True
        hwp.HAction.Run("Cancel")
    return False


try:
    hwp.Open(src, "", "")
    # docx 에서 온 문서는 'MS Word 호환' 조판이라 줄 간격 160 % 가 훨씬 넓게 나온다.
    ps = hwp.HParameterSet.HCompatibleDocument
    hwp.HAction.GetDefault("CompatibleDocument", ps.HSet)
    ps.TargetProgram = 0
    hwp.HAction.Execute("CompatibleDocument", ps.HSet)

    tables = []
    c = hwp.HeadCtrl
    while c:
        if c.CtrlID == "tbl":
            tables.append(c)
        c = c.Next
    bad = rows = 0
    for c in tables[SKIP:]:
        if not enter(c):
            bad += 1
            continue
        hwp.HAction.Run("TableCellBlockExtend")
        hwp.HAction.Run("TableCellBlockExtend")
        act = hwp.CreateAction("ParagraphShape")
        pset = act.CreateSet()
        pset.SetItem("LineSpacingType", 0)
        pset.SetItem("LineSpacing", CELL_SPACING)
        act.Execute(pset)
        hwp.HAction.Run("Cancel")
        hwp.HAction.Run("MoveDocBegin")

        enter(c)
        hwp.HAction.Run("TableCellBlockExtend")
        hwp.HAction.Run("TableCellBlockExtend")
        so = hwp.HParameterSet.HShapeObject
        hwp.HAction.GetDefault("TablePropertyDialog", so.HSet)
        so.HSet.SetItem("ShapeType", 3)
        so.ShapeTableCell.VertAlign = 1
        hwp.HAction.Execute("TablePropertyDialog", so.HSet)
        hwp.HAction.Run("Cancel")
        hwp.HAction.Run("MoveDocBegin")

        enter(c)
        hwp.HAction.Run("Cancel")
        prev = None
        while hwp.GetPos() != prev:
            prev = hwp.GetPos()
            h = hwp.CellShape.Item("Cell").Item("Height")
            hwp.HAction.Run("TableCellBlock")
            so = hwp.HParameterSet.HShapeObject
            hwp.HAction.GetDefault("TablePropertyDialog", so.HSet)
            so.HSet.SetItem("ShapeType", 3)
            so.HSet.SetItem("ShapeCellSize", 1)
            so.ShapeTableCell.Height = h + ROW_PAD
            hwp.HAction.Execute("TablePropertyDialog", so.HSet)
            hwp.HAction.Run("Cancel")
            rows += 1
            hwp.HAction.Run("TableLowerCell")
        hwp.HAction.Run("MoveDocBegin")

        # 바깥 여백은 행 높이를 다 맞춘 뒤, 방금 읽은 속성에 얹어 준다.
        # 오래된 속성을 되돌려 쓰면 표 높이가 예전 값으로 돌아가며 행이 늘어난다.
        pr = c.Properties
        pr.SetItem("OutsideMarginTop", 425)
        pr.SetItem("OutsideMarginBottom", 850)
        c.Properties = pr
    print("tables", len(tables), "not selected", bad, "rows", rows)
    print("save", hwp.SaveAs(out, "HWP", ""), hwp.SaveAs(pdf, "PDF", ""), "pages", hwp.PageCount)
finally:
    hwp.Clear(1)
    hwp.Quit()
