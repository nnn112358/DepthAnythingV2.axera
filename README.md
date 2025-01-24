# DepthAnythingV2.axera

![image](https://github.com/user-attachments/assets/50b75567-c8ef-46d5-8df9-c2dba6b7e730)
https://github.com/nnn112358/DepthAnythingV2.axera/blob/main/python/infer.py
![image](https://github.com/user-attachments/assets/bdfe6dd3-18bc-4ea2-915a-1141acb36034)
https://github.com/nnn112358/DepthAnythingV2.axera/blob/main/python/depth_anything_v2_ax_camera_stream_FastAPI.py

# DepthAnythingV2.axera

Axera上のDepthAnything v2デモ
- C++とPythonの2言語に対応。C++コードは[ax-samples](https://github.com/AXERA-TECH/ax-samples/blob/main/examples/ax650/ax_depth_anything_steps.cc)にあります
- 事前コンパイル済みモデルは[models](https://github.com/AXERA-TECH/DepthAnythingV2.axera/releases/download/v1.0.0/models.tar.gz)からダウンロード。自身で変換する場合は[モデル変換](/model_convert/README.md)を参照

## 対応プラットフォーム
- [x] AX650N
- [x] AX630C

## モデル変換
[モデル変換](./model_convert/README.md)

## ボード実装
- AX650NデバイスにはUbuntu22.04がプリインストール
- AX650Nボードにrootでログイン 
- インターネットに接続し、`apt install`や`pip install`などのコマンドが実行可能なことを確認
- 動作確認済みデバイス：AX650N DEMOボード、AiPi Pro(AX650N)、AiPi 2(AX630C)

### Python API実行
#### 必要条件
```
mkdir /opt/site-packages
cd python
pip3 install -r requirements.txt --prefix=/opt/site-packages
```

#### 環境変数の追加
以下2行を`/root/.bashrc`に追加(実際のパスは要確認)し、ターミナルを再接続または`source ~/.bashrc`を実行
```
export PYTHONPATH=$PYTHONPATH:/opt/site-packages/local/lib/python3.10/dist-packages
export PATH=$PATH:/opt/site-packages/local/bin
```

#### 実行方法
##### ONNX Runtimeベースの実行
開発ボードまたはPC上で実行可能

以下のコマンドを実行:
```
cd python
python3 infer_onnx.py --img examples/demo01.jpg --model ../models/depth_anything_v2_vits.onnx
```

##### AXEngineベースの実行
開発ボード上で以下のコマンドを実行:
```
cd python
python3 infer.py --img examples/demo01.jpg --model ../models/compiled.axmodel
```

実行パラメータ:
| パラメータ名 | 説明 |
|---|---|
| --img | 入力画像パス |
| --model | モデルパス |

### レイテンシ
#### AX650N
| モデル | レイテンシ(ms) |
|---|---|
|depth_anyting_v2_vits|33.1|

#### AX630C
| モデル | レイテンシ(ms) |
|---|---|
|depth_anyting_v2_vits|300.1|

## 技術討論
- Githubのissues
- QQグループ: 139953715
