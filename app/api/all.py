from src.recognition import YarnPredictor, YarnModel


g_predictor = YarnPredictor(YarnModel(model_name='yarn_sim2',out_dim=4))
