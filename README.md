# 1. Install dependencies
```
pip install torch transformers vncorenlp scikit-learn tqdm
```
# 2. Download vnCoreNLP
```
mkdir VnCoreNLP
cd VnCoreNLP
wget https://github.com/vncorenlp/VnCoreNLP/archive/master.zip
unzip master.zip
```
# Download VnCoreNLP-1.1.1.jar from the repository

# Augment data

python augmentation.py --input train.jsonl --output train_aug.jsonl --ratio 0.3

# 4. Train model
```
python train.py --train_file train_aug.jsonl --vncorenlp_path VnCoreNLP/VnCoreNLP-1.1.1.jar
```
# 5. Test model
```
python test.py --test_file test.jsonl --model_path best_model.pt --vncorenlp_path VnCoreNLP/VnCoreNLP-1.1.1.jar
```