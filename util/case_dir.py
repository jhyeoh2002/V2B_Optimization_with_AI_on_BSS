import os
import config as cfg

def case_dir(base_dir: str, case_id: int) -> str:
    case_dirs = {
        0: os.path.join(base_dir, cfg.CASE0),
        1: os.path.join(base_dir, cfg.CASE1),
        2: os.path.join(base_dir, cfg.CASE2),
        3: os.path.join(base_dir, cfg.CASE3),
    }
    return case_dirs.get(case_id, None)