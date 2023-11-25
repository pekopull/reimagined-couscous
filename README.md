
# README

初賽期間我使用的環境是 Google Colab Pro，使用的是 Jupyter Notebook (.ipynb) 檔案的環境。
因此我並不熟悉如何使用 .py 的環境，所以可能會有一些奇怪的地方，請見諒。
建議參考 .ipynb 檔案，並且使用 Jupyter Notebook 執行。

## Preprocess

初賽期間我並沒有把處理好的資料儲存下來，因為檔案太大了，所以每次執行都會重新處理一次資料(非常沒有效率)。
未來如果要改進的話，首先就是要把經過特徵工程處理好的資料儲存下來，然後再進行訓練。
另外我認為我在特徵工程的部分做得還不夠好。
可能有些特徵對模型來說是相對沒那麼重要的，但是由於比賽需要盡量提高分數，所以即使是沒那麼重要的特徵也會保留下來。
也許未來可以嘗試移除不必要的特徵，或是使用其他的特徵工程方法。

## Model

使用 Catboost 作為模型，並且使用 GPU 訓練。
雖然 Catboost 也可以使用 CPU 訓練，但是使用 GPU 訓練可以大幅提升訓練速度。
訓練環境為 Google Colab Pro，使用的是免費的 T4 GPU，並不需要使用到付費 GPU。
使用 Google Colab 的 Pro 版本單純只是因為記憶體不足。
訓練一次模型大約需要 3~5 分鐘，在 Preprocess 階段反而會花費較多時間(約20分鐘)。



# 檔案說明與執行方法


## 資料夾結構

Preprocess/: 存放前處理的code
Model/: 存放模型訓練相關code
Model/saved_models: 存放訓練好的 Catboost 模型 (.cbm)
requirements.txt: 需要的套件
run_train.py: 使用所有資料進行訓練並儲存模型
run_validation.py: 使用一部分資料進行訓練，一部分資料進行驗證，不會儲存模型
run_inference.py: 使用訓練好的 .cbm 模型檔案進行 Inference


## 執行方法

安裝所需套件
```
$ pip install -r requirements.txt
```

### run_inference.py

根據 Catboost 官方文件 
(<https://catboost.ai/en/docs/features/training-on-gpu>)
> Training on GPU is non-deterministic, because the order of floating point summations is non-deterministic in this implementation.

使用 GPU 訓練模型會有非確定性，因此每次訓練結果會有些微差異。
因此如果需要取得一模一樣的測試分數，需要使用 .cbm 模型檔案進行 Inference。

run_inference.py 會執行 Preprocess/load.py 以及 Preprocess/preprocess.py，然後在 Model/inference.py 載入 .cbm 模型檔案進行 Inference。最後產生繳交的.csv檔案。


執行參數
- ${1}: 訓練資料路徑
- ${2}: 測試資料路徑
- ${3}: 區分訓練資料與測試資料的日期 (locdt)
- ${4}: 區分訓練資料與驗證資料的日期 (locdt)
- ${5}: 模型讀取路徑 (.cbm)
- ${6}: 輸出檔案路徑 (.csv)

執行範例
```
$ python run_inference.py dataset_1st/training.csv dataset_2nd/public.csv 60 56 Model/saved_models/catboost_iter1146.cbm output.csv
```

### run_train.py

如需訓練模型，可以使用 run_train.py 進行訓練。
run_train.py 會執行 Preprocess/load.py 以及 Preprocess/preprocess.py，然後在 Model/train.py 訓練模型並儲存模型。最後產生繳交的.csv檔案。
請注意由於使用 GPU 訓練模型會有非確定性，因此每次訓練結果會有些微差異。


執行參數
- ${1}: 訓練資料路徑
- ${2}: 測試資料路徑
- ${3}: 區分訓練資料與測試資料的日期 (locdt)
- ${4}: 區分訓練資料與驗證資料的日期 (locdt)
- ${5}: 模型儲存路徑 (.cbm)
- ${6}: 輸出檔案路徑 (.csv)

執行範例
```
$ python run_train.py dataset_1st/training.csv dataset_2nd/public.csv 60 56 Model/saved_models/model.cbm output.csv
```

### run_validation.py

如需使用一部分資料驗證模型，可以使用 run_validation.py 進行驗證。
會根據 ${4} 的日期將資料分成訓練資料與驗證資料，然後使用訓練資料進行訓練，使用驗證資料進行驗證。
run_validation.py 會執行 Preprocess/load.py 以及 Preprocess/preprocess.py，然後在 Model/train.py 訓練模型並儲存模型。

執行參數
- ${1}: 訓練資料路徑
- ${2}: 測試資料路徑
- ${3}: 區分訓練資料與測試資料的日期 (locdt)
- ${4}: 區分訓練資料與驗證資料的日期 (locdt)

執行範例
```
$ python run_validation.py dataset_1st/training.csv dataset_2nd/public.csv 60 56
```

## ipynb 檔案

如果 .py 檔案執行有問題，或是想要更了解程式碼的話。
建議參考 aicup2023.ipynb 檔案。與.py的環境相比，裡面額外包含了 Adversarial Validation 以及 Data Explore 兩個部分。
Adversarial Validation (對抗驗證) 是一種檢查訓練資料與測試資料是否有差異的方法。 
可以幫助我們找出最需要檢查的特徵、確認資料是否正確處理、找出和測試集分布類似的驗證資料

- 如果要進行 Adversarial Validation (對抗驗證)，可以在執行 # Feature Engineering 之前或之後執行 # Adversarial Validation

- 如果要訓練模型然後使用一部分資料進行驗證，不需要執行 # Adversarial Validation，並且請跳過 # Data Explore
在執行完 # Feature Engineering 之後，執行 # Catboost Training with validation

- 如果要直接使用全部資料訓練模型，請跳過 # Adversarial Validation 以及 # Data Explore，
在執行完 # Feature Engineering 之後，執行 # Catboost Train with all data


# 超參數

catboost_iter1146.cbm 是使用 dataset_1st/training.csv 以及 dataset_2nd/public.csv 訓練的模型
以下為 catboost_iter1146.cbm 使用的超參數，訓練時間花費 213 秒 

- 'iterations': 1200
- 'eval_metric': 'F1'
- 'task_type': 'GPU'
- 'early_stopping_rounds': 200
- 'verbose': 20
- 'learning_rate': 0.054
- 'has_time':True
- 'depth':6
- 'grow_policy':"Lossguide"
- 'class_weights':{0: 2.0, 1: 1.0}
- 'boosting_type': "Plain"



# Future Works

經過特徵工程會產生很多針對使用者或是商店的特徵，這些特徵使用表格來儲存並不是最好的做法。也許一個更好的方式是使用 Graph 這種網路狀的結構來儲存資料。將使用者以及商店用點(Node)來表示，交易使用線(Edge)來表示。最後使用 Graph Neural Network (GNN) 來做訓練。

可參考以下資料
- <https://martin12345m.medium.com/%E5%9C%96%E5%83%8F%E5%8C%96%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF-1-graph-neural-network%E5%B0%8F%E7%B0%A1%E4%BB%8B-82c7a5d843c1>
- <https://paperswithcode.com/task/fraud-detection>