About YarnCV 

0. acquisition
	- camera sampling at fps=30

1. Yarn Dataset
	- directory: "data/refined-2";
	- num classes=4: 0=normal, 1~4=abnormal. *Note: label-3 is sikpped since it depends heavily on real-time grab;
	- train/test spliting: see `write2csv.py`;

2. model, ckpt and metrics
	- modified resnet18: ckpt/efficientnet-100.00, acc=47
	- modified efficientnet-b0: ckpt/resnet-100.00.pth, acc=54
	- yarn_sim: ckpt/yarn_sim2-100.00.pth, acc=76.62
	- yarn_sim2: ckpt/yarn_sim2-100.00.pth, acc=100.0

3. challenge 
	- test model on another testset "data/img-test" (different visual field from "data/refined-2")
	- performance of model yarn_sim2: ~50 acc

4. another attempt: visual large model 
	- prompt engineering: 
		- Qwen-vl-max: ~75 acc
	- fine-tuning (future work)
		- peft + LoRA 
		- http api: see bailian.console.aliyun.com

5. related resources (urls)
	- data: see data/.gitkeep
	- ckpt: see src/recognition/out/ckpt/.gitkeep




