About YarnCV 

0. acquisition
	- camera sampling at fps=30

1. Yarn Dataset
	- directory: "data/refined" and "data/refined-2/"; *Note: label-3 is sikpped since it depends heavily on real-time*;
	- num classes=4: 0=normal, 1~4=abnormal. 

2. model, ckpt and metrics
	- models: modified resnet18, modified efficientnet-b0, yarn_sim, yarn_sim2 (best acc=100)
	- ckpt: see "out/ckpt/"
	- metrics: accuracy

3. another attempt: visual large model 
	- challenge: yarn_sim2 has poor acc (~50) on img-test 
	- prompt engineering: 
		- Qwen-vl-max: ~75 acc
	- future work: supervised fine-tuning 
		- method: peft + LoRA 
		- useful platforms: bailian.console.aliyun.com

4. related resources (urls)
	- data: see data/.gitkeep
	- ckpt: see src/recognition/out/ckpt/.gitkeep