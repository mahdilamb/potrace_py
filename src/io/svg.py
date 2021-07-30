from typing import List, Union

from src.common import Writer, Path, Curve, SegmentTag, Bitmap


class SVGWriter(Writer):
    file_ext = ".svg"

    @classmethod
    def _get_svg(cls, bm: Bitmap, pathlist: List[Path], size: float = 1., opt_type: str = None,
                 color: str = "black") -> str:
        def to_fixed(x: float, dp: int = 3) -> str:
            return str(round(x, dp))

        def path(curve: Curve) -> str:
            def cubic_bezier(curve: Curve, i: int) -> str:
                b = 'C ' + to_fixed(curve.c[i, 0][0] * size) + ' ' + to_fixed(curve.c[i, 0][1] * size) + ','
                b += to_fixed(curve.c[i, 1][0] * size) + ' ' + to_fixed(curve.c[i, 1][1] * size) + ','
                b += to_fixed(curve.c[i, 2][0] * size) + ' ' + to_fixed(curve.c[i, 2][1] * size) + ' '
                return b

            def segment(curve: Curve, i: int) -> str:
                s = 'L ' + to_fixed(curve.c[i, 1][0] * size) + ' ' + to_fixed(curve.c[i, 1][1] * size) + ' '
                s += to_fixed(curve.c[i, 2][0] * size) + ' ' + to_fixed(curve.c[i, 2][1] * size) + ' '
                return s

            n = curve.n
            p = 'M' + to_fixed(curve.c[(n - 1), 2][0] * size) + ' ' + to_fixed(curve.c[(n - 1), 2][1] * size) + ' '
            p += "".join([cubic_bezier(curve, i) if curve.tag[i] == SegmentTag.CURVE_TO else segment(curve, i) for i in
                          range(n)])
            return p

        w: str = str(bm.w * size)
        h: str = str(bm.h * size)
        svg = '<svg id="svg" version="1.1" width="' + w + '" height="' + h + '" xmlns="http://www.w3.org/2000/svg">'
        svg += '<path d="'
        svg += "".join([path(p.curve) for p in pathlist])
        strokec: str
        fillc: str
        fillrule: str
        if opt_type == "curve":
            strokec = color
            fillc = "none"
            fillrule = ''
        else:
            strokec = "none"
            fillc = color
            fillrule = ' fill-rule="evenodd"'
        svg += '" stroke="' + strokec + '" fill="' + fillc + '"' + fillrule + '/></svg>'
        return svg

    @staticmethod
    def write(bm: Bitmap, pathlist: List[Path], output: Union[str, Path], **kwargs):
        size: float = kwargs.get("size", 1.)
        with open(output, 'w') as f:
            f.write(SVGWriter._get_svg(bm, pathlist, size=size, opt_type=kwargs.get("opt_type", None),
                                       color=kwargs.get("color", "black")))
