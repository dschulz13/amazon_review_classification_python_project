import numpy as np

with open("README.md", "r", encoding = "utf-8") as file:
    lines = file.readlines()


lines = np.array(lines)
w_summary = (lines == '<summary>\n')
line_idx = np.array(range(0, len(lines)))
line_gap = line_idx[w_summary] + 1

idx = np.array([True] * len(lines))
idx[line_gap] = False

lines_upd = lines[idx]
lines_upd = lines_upd.tolist()

with open("README.md", "w", encoding = "utf-8") as file:
    file.writelines(line for line in lines_upd)

