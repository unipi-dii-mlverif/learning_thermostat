import os

from .config import T_desired, LL, UL, C_in, H_in, T_derivative, time_since_comm


def export_tex_pgf(result_matrix, x_grid, x_step, y_grid, y_step,
                   xlabel, ylabel, title, tex_path, subtitle=None):
    """Write a standalone TikZ/pgfplots TeX file with B/W pattern fills.

    result_matrix[i, j]: 0 = OFF (white), 1 = possibly ON (stripes), 2 = ON (crosshatch)
    x_grid[j] / y_grid[i] are the lower edges of each cell.
    """
    nrows, ncols = result_matrix.shape

    # Horizontal run-length merge: consecutive same-value cells in each row
    # become a single rectangle, dramatically reducing the number of \fill commands.
    rects = {0: [], 1: [], 2: []}
    for i in range(nrows):
        j = 0
        while j < ncols:
            val = int(result_matrix[i, j])
            j_start = j
            while j < ncols and int(result_matrix[i, j]) == val:
                j += 1
            rects[val].append((
                float(x_grid[j_start]),
                float(x_grid[j - 1]) + x_step,
                float(y_grid[i]),
                float(y_grid[i]) + y_step,
            ))

    def fill_cmds(rect_list, style):
        return "\n".join(
            "  \\fill[" + style + "]"
            " (axis cs:" + f"{x0:.5g},{y0:.5g}" + ")"
            " rectangle (axis cs:" + f"{x1:.5g},{y1:.5g}" + ");"
            for x0, x1, y0, y1 in rect_list
        )

    xmin = float(x_grid[0])
    xmax = float(x_grid[-1]) + x_step
    ymin = float(y_grid[0])
    ymax = float(y_grid[-1]) + y_step

    maybe_body = fill_cmds(rects[1], "pattern=north east lines, pattern color=black")
    on_body    = fill_cmds(rects[2], "pattern=crosshatch, pattern color=black")

    # ── Boundary edge detection (run-length merged) ───────────────────────────
    bsegs = {(0, 1): [], (0, 2): [], (1, 2): []}
    # Horizontal boundaries (between row i and row i+1)
    for i in range(nrows - 1):
        j = 0
        while j < ncols:
            v_b, v_t = int(result_matrix[i, j]), int(result_matrix[i + 1, j])
            if v_b != v_t:
                kind = (min(v_b, v_t), max(v_b, v_t))
                j_start = j
                while j < ncols and int(result_matrix[i, j]) == v_b and int(result_matrix[i + 1, j]) == v_t:
                    j += 1
                y = float(y_grid[i]) + y_step
                bsegs[kind].append((float(x_grid[j_start]), y, float(x_grid[j - 1]) + x_step, y))
            else:
                j += 1
    # Vertical boundaries (between col j and col j+1)
    for j in range(ncols - 1):
        i = 0
        while i < nrows:
            v_l, v_r = int(result_matrix[i, j]), int(result_matrix[i, j + 1])
            if v_l != v_r:
                kind = (min(v_l, v_r), max(v_l, v_r))
                i_start = i
                while i < nrows and int(result_matrix[i, j]) == v_l and int(result_matrix[i, j + 1]) == v_r:
                    i += 1
                x = float(x_grid[j]) + x_step
                bsegs[kind].append((x, float(y_grid[i_start]), x, float(y_grid[i - 1]) + y_step))
            else:
                i += 1

    def draw_cmds(seg_list, style):
        return "\n".join(
            "  \\draw[" + style + "]"
            " (axis cs:" + f"{x0:.5g},{y0:.5g}" + ")"
            " -- (axis cs:" + f"{x1:.5g},{y1:.5g}" + ");"
            for x0, y0, x1, y1 in seg_list
        )

    border_01 = draw_cmds(bsegs[(0, 1)], "black, thick")
    border_12 = draw_cmds(bsegs[(1, 2)], "black, thick")
    border_02 = draw_cmds(bsegs[(0, 2)], "black, thick")

    lines = [
        r"\documentclass{standalone}",
        r"\usepackage{pgfplots}",
        r"\usetikzlibrary{patterns}",
        r"\pgfplotsset{compat=1.18}",
        "",
        r"\begin{document}",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"  width=4.33in,",
        r"  height=2.9in,",
        f"  xlabel={{{xlabel}}},",
        f"  ylabel={{{ylabel}}},",
        (f"  title={{\\textbf{{{title}}}\\\\{{\\small {subtitle}}}}},"
         if subtitle else f"  title={{{title}}},"),
        r"  title style={align=center},",
        f"  xmin={xmin:.5g}, xmax={xmax:.5g},",
        f"  ymin={ymin:.5g}, ymax={ymax:.5g},",
        r"  axis background/.style={fill=white},",
        r"  legend style={at={(0.02,0.02)}, anchor=south west,",
        r"                 font=\small, cells={align=left}},",
        r"  tick align=outside,",
        r"  grid=both, grid style={line width=0.2pt, draw=gray!30},",
        r"]",
        r"  %% Possibly ON -- diagonal stripes",
        maybe_body,
        r"  %% Definitely ON -- crosshatch",
        on_body,
        r"  %% Region boundaries (dashed=OFF/Maybe, solid=Maybe/ON, dotted=OFF/ON)",
        border_01,
        border_12,
        border_02,
        r"  %% Legend",
        r"  \addlegendimage{area legend, fill=white, draw=black}",
        r"  \addlegendentry{Definitely OFF}",
        r"  \addlegendimage{area legend, pattern=north east lines,"
        r" pattern color=black, draw=black}",
        r"  \addlegendentry{Possibly ON}",
        r"  \addlegendimage{area legend, pattern=crosshatch,"
        r" pattern color=black, draw=black}",
        r"  \addlegendentry{Definitely ON}",
        r"\end{axis}",
        r"\end{tikzpicture}",
        r"\end{document}",
    ]

    with open(tex_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"TeX file written to {tex_path}")


def export_tex_interval_plot(temperature_intervals, out_intervals, tex_path):
    """Write a standalone TikZ/pgfplots TeX file for the interval bound plot.

    Produces a filled-area plot (lower/upper bound shaded band) with a dashed
    threshold line at y=0.5, B/W friendly: the band uses north-east-lines
    pattern, bounds are solid black lines.
    """
    means   = [sum(ti) / 2 for ti in temperature_intervals]
    lowers  = [iv[0] for iv in out_intervals]
    uppers  = [iv[1] for iv in out_intervals]

    def coords(xs, ys):
        return " ".join(f"({x:.4g},{y:.6g})" for x, y in zip(xs, ys))

    lower_coords = coords(means, lowers)
    upper_coords = coords(means, uppers)
    # fillbetween needs the upper curve traced in reverse to close the polygon
    upper_rev_coords = coords(reversed(means), reversed(uppers))

    xmin = means[0]  - 0.5
    xmax = means[-1] + 0.5

    subtitle = (
        f"$T_{{\\mathrm{{des}}}}={T_desired}$, "
        f"$\\mathrm{{LL}}={LL}$, $\\mathrm{{UL}}={UL}$, "
        f"$C_{{\\mathrm{{in}}}}={C_in}$\\,s, $H_{{\\mathrm{{in}}}}={H_in}$\\,s, "
        f"$\\dot{{T}}={T_derivative}$\\,$^{{\\circ}}$C/s, "
        f"$t_{{\\mathrm{{com}}}}={time_since_comm}$\\,s"
    )

    lines = [
        r"\documentclass{standalone}",
        r"\usepackage{pgfplots}",
        r"\usepackage{pgfplotstable}",
        r"\usetikzlibrary{patterns}",
        r"\pgfplotsset{compat=1.18}",
        "",
        r"\begin{document}",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"  width=4.33in,",
        r"  height=2.2in,",
        r"  xlabel={Temperature Intervals ($^{\circ}$C)},",
        r"  ylabel={Model Output},",
        f"  title={{\\textbf{{Output Intervals vs Temperature}}\\\\{{\\small {subtitle}}}}},",
        r"  title style={align=center},",
        f"  xmin={xmin:.4g}, xmax={xmax:.4g},",
        r"  ymin=0, ymax=1,",
        r"  ytick={0, 0.5, 1},",
        r"  xtick={",
        "    " + ", ".join(f"{m:.4g}" for m in means),
        r"  },",
        r"  xticklabels={",
        "    " + ", ".join(f"{{$[{ti[0]},{ti[1]}]$}}" for ti in temperature_intervals),
        r"  },",
        r"  x tick label style={rotate=45, anchor=east, font=\tiny},",
        r"  legend style={at={(0.02,0.02)}, anchor=south west, font=\scriptsize},",
        r"  tick align=outside,",
        r"  grid=both, grid style={line width=0.2pt, draw=gray!30},",
        r"]",
        r"  %% Shaded interval band (north-east-lines pattern)",
        r"  \addplot[",
        r"    pattern=north east lines, pattern color=black,",
        r"    draw=none,",
        r"  ] coordinates {",
        r"    %% lower edge forward, upper edge backward to close polygon",
        f"    {lower_coords}",
        f"    {upper_rev_coords}",
        r"  } -- cycle;",
        r"  %% Lower bound",
        r"  \addplot[black, thick] coordinates { " + lower_coords + r" };",
        r"  \addlegendentry{Lower Bound}",
        r"  %% Upper bound",
        r"  \addplot[black, thick, dashed] coordinates { " + upper_coords + r" };",
        r"  \addlegendentry{Upper Bound}",
        r"  %% Threshold line",
        r"  \addplot[black, dotted, thick, domain=" + f"{xmin:.4g}:{xmax:.4g}" + r"] {0.5};",
        r"  \addlegendentry{Threshold $0.5$}",
        r"\end{axis}",
        r"\end{tikzpicture}",
        r"\end{document}",
    ]

    with open(tex_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"TeX file written to {tex_path}")
