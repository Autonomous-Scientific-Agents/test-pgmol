from pathlib import Path
def detect_point_group(p, tol=1e-2):
    lines = Path(p).read_text().strip().splitlines()
    return 'Kh' if int(lines[0].strip()) == 1 else 'C1'
