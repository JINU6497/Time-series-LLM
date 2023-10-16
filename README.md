# Time-series-LLM
Time-series framework for LLM

Method
======
LLM4TS
------
![image](https://github.com/JINU6497/Time-series-LLM/assets/87464956/0df419f8-ec29-430f-bae6-db811f8e5b68)

- PLM에 어떻게 Time-series data가 가지는 특성을 전달할 수 있을까?
  - 2 Stage Fine-tuning (Supervised, Downstream)
- Time-series data를 PLM에 어떤 형태의 Input으로 넣을 수 있을까?
  - Patching + Channel Independence
- PLM을 어떤 방식으로 Fine-tuning할 수 있을까?
  - Layer Normalization tuning + Low-Rank Adaptation

# 1. 2 Stage Fine-tuning (supervised, Donwstream)
## Supervised Fine-tuning (SFT)
- PLM은 대량의 Corpus로 Pre-train 되었지만, 사실 Corpus data(Natural Language)와 Time-series data 간에는 상당한 차이점이 존재하기에, PLM에 Time-series data의 특성을 Align 시켜주는 과정 필요
- Patch 단위를 Autoregressive하게 생성하도록 하며 진행
## Downstream Fine-tuning (DFT)
- Supervised fine-tuning 과정을 통하여 PLM과 Time-series data가 Align 되었다면, Backbone model의 weight를  Forecasting이라는 특정 Task 잘 수행할 수 있도록 Downstream fine-tuning 진행
- Final layer만을 학습시키는 Linear probing을 절반 수행 후, Full fine-tuning진행

# 2. Patching + Channel Independence
## Instance Normalization
- SFT 과정에서는 Instance Normalization만을, DFT 과정에서는 RevIN 사용
- Feature 단위는 고려하지 않고 인접한 Time-step만을 단일 Patch base의 Token으로 사용
## Three Encoding
- Token Encoding: Patched 된 Time-series data token은 Time-series data이며, Scalar data가 아닌 Vector 형태이기에 1D Convolution 이용
- Positional Encoding: Learnable한 Look-up table 이용하여 Patch 위치 Mapping
- Temporal Encoding: Patch 단위로 데이터를 변경하기에, Patch의 맨 앞 데이터의 시간 정보를 Patch의 대표 정보로 이용

# 3. Pre-trained LLM and PEFT
- Layer normalization tuning 및 LoRA이용

# 4. Output layer
- SFT 단계에서는 Patched된 Time-series data의 형태 그대로를 유지하도록 Output 다오게 함
- DFT 단계에서는 Patching된 Time-series Output을 Linear layer 이후 Flattening한 후, Rearrange함으류써 일반 시계열 형태로 만들어줌 

Experiments
===========
|               |     | DLinear    | LLM4TS     | TimesNet | PatchTST | DLinear    | LLM4TS     | TimesNet | PatchTST   |
|---------------|-----|------------|------------|----------|----------|------------|------------|----------|------------|
|               |     | MSE        |            |          |          | MAE        |            |          |            |
| ETTh1         | 96  | 0.4822     | **0.4554** | 1.1943   | 0.5616   | 0.4835     | **0.4617** | 0.8369   | 0.5281     |
|               | 192 | 0.5310     | **0.5156** | 1.1693   | 0.6202   | 0.5180     | **0.5006** | 0.8248   | 0.5615     |
|               | 336 | 0.5745     | **0.5590** | 1.1868   | 0.6815   | 0.5491     | **0.5283** | 0.8328   | 0.5945     |
|               | 720 | **0.6826** | 0.7215     |          | 0.8395   | **0.6165** | 0.6197     |          | 0.6802     |
|               | AVG | 0.5676     | **0.5628** | 1.1835   | 0.6757   | 0.5418     | **0.5276** | 0.8315   | 0.5911     |
| ETTm1         | 96  | 0.3618     | **0.3521** | 0.7299   | 0.3657   | 0.3926     | **0.3824** | 0.6468   | 0.3940     |
|               | 192 | 0.4279     | **0.4251** | 0.7717   | 0.4398   | 0.4276     | **0.4211** | 0.6657   | 0.4330     |
|               | 336 | **0.4922** | 0.4973     | 1.1126   | 0.5139   | 0.4640     | **0.4611** | 0.7802   | 0.4732     |
|               | 720 | **0.5553** | 0.5796     | 1.0976   | 0.5912   | **0.5104** | 0.5144     | 0.7848   | 0.5229     |
|               | AVG | **0.4593** | 0.4635     | 0.9280   | 0.4776   | 0.4487     | **0.4447** | 0.7194   | 0.4558     |
| weather       | 96  | 0.2015     | **0.1988** | 0.7363   | 0.2000   | 0.2680     | **0.2364** | 0.6608   | 0.2413     |
|               | 192 | **0.2415** | 0.2431     | 0.7897   | 0.2443   | 0.3051     | **0.2721** | 0.6888   | 0.2757     |
|               | 336 | **0.2860** | 0.2920     |          | 0.2946   | 0.3397     | **0.3070** |          | 0.3099     |
|               | 720 | **0.3496** | 0.3616     | 0.8158   | 0.3660   | 0.3889     | **0.3515** | 0.6966   | 0.3552     |
|               | AVG | **0.2697** | 0.2739     | 0.7806   | 0.2762   | 0.3254     | **0.2918** | 0.6821   | 0.2955     |
| exchange_rate | 96  | 0.1229     | **0.0853** | 2.5827   | 0.0885   | 0.2685     | **0.2095** | 1.3330   | 0.2146     |
|               | 192 | 0.2008     | **0.1949** |          | 0.1734   | 0.3478     | 0.3221     |          | **0.3052** |
|               | 336 | **0.2899** | 0.3579     |          | 0.3197   | **0.4162** | 0.4423     |          | 0.4166     |
|               | AVG | **0.2045** | 0.2127     | 2.5827   | 0.1939   | 0.3441     | 0.3246     | 1.3330   | **0.3121** |
