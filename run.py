import datetime as dt
import pathlib

import sh


nbx = sh.jupyter.nbconvert.bake(
    execute=True, to="notebook", inplace=True
).bake("--ExecutePreprocessor.timeout=600")


FILES = [
    '00_section_features_evaluation.ipynb'
    '01_section_model_best_params.ipynb'
    '02_section_model_selection.ipynb'
    '03_section_analysis_model_selection.ipynb'
    '04_section_unbalance_same_size.ipynb'
    '05_section_unbalance_diff_size.ipynb'
    '06_unbalance_diff_analysis.ipynb'
]


def run():
    for path in FILES:
        print(f"[{dt.datetime.now().isoformat()}] Executing {path}")
        
        #nbx(path)
        
        
    
if __name__ == "__main__":
    run()