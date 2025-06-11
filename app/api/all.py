from src.recognition import YarnPredictor
from src.vlm import QwenPredictor 

# local
g_predictor = YarnPredictor('yarn_sim2',out_dim=4)

# openai - qwen
# [X] qwen-vl-max
# [O] qwen-vl-max-2025-04-02,qwen-vl-max-1230,qwen-vl-max-2025-01-25,qwen-vl-max-1119,qwen-vl-max-1030,qwen-vl-max-0809
# [O] qwen-vl-plus,qwen-vl-plus-2025-05-07,qwen-vl-plus-2025-01-25,qwen-vl-plus-0102,qwen-vl-plus-0809
qwen_predicator = QwenPredictor('qwen-vl-max-2025-04-08')  # 'qwen-vl-max-2025-01-25'