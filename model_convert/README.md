# モデル変換

## 仮想環境の作成
```
conda create -n depthanythingv2 python=3.12 -y
conda activate depthanythingv2
```

## 依存関係のインストール
```
pip install -r requirements.txt
```

## モデルのエクスポート（PyTorch -> ONNX）
smallモデルのエクスポート:
```
python export_onnx.py --encoder vits --onnx-path  depth_anything_v2_vits.onnx
```

成功すると`depth_anything_v2_vits.onnx`が生成されます。

## モデル変換（ONNX -> Axera）
`Pulsar2`ツールを使用してONNXモデルをAxera NPU用の`.axmodel`形式に変換:

1. モデル用のPTQ量子化キャリブレーションデータセットの生成
2. `Pulsar2 build`コマンドでモデル変換(PTQ量子化、コンパイル)
詳細は[AXera Pulsar2ツールチェーンガイド](https://pulsar2-docs.readthedocs.io/zh-cn/latest/index.html)を参照

### 量子化データセットのダウンロード
```
bash download_dataset.sh
```
単一画像入力の簡単なモデルなので、パッケージ化された画像データを直接ダウンロード

### モデル変換
#### 設定ファイルの修正
`config.json`の`calibration_dataset`フィールドを、ダウンロードした量子化データセットのパスに変更

#### Pulsar2 buildの実行
```
pulsar2 build --input depth_anything_v2_vits.onnx --config config.json --output_dir build-output --output_name depth_anything_v2_vits.axmodel --target_hardware AX650 --compiler.check 0
```
