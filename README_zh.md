# DepthAnythingV2.axera
DepthAnything v2 DEMO on Axera

- 目前支持 C++ 和 Python 两种语言，C++代码在 [ax-samples](https://github.com/AXERA-TECH/ax-samples/blob/main/examples/ax650/ax_depth_anything_steps.cc)
- 预编译模型下载[models](https://github.com/AXERA-TECH/DepthAnythingV2.axera/releases/download/v1.0.0/models.tar.gz)，如需自行转换请参考[模型转换](/model_convert/README.md)

## 支持平台

- [x] AX650N
- [x] AX630C

## 模型转换

[模型转换](./model_convert/README.md)

## 上板部署

- AX650N 的设备已预装 Ubuntu22.04
- 以 root 权限登陆 AX650N 的板卡设备
- 链接互联网，确保 AX650N 的设备能正常执行 `apt install`, `pip install` 等指令
- 已验证设备：AX650N DEMO Board、爱芯派Pro(AX650N)、爱芯派2(AX630C)

### Python API 运行

#### Requirements

```
mkdir /opt/site-packages
cd python
pip3 install -r requirements.txt --prefix=/opt/site-packages
``` 

#### 添加环境变量

将以下两行添加到 `/root/.bashrc`(实际添加的路径需要自行检查)后，重新连接终端或者执行 `source ~/.bashrc`

```
export PYTHONPATH=$PYTHONPATH:/opt/site-packages/local/lib/python3.10/dist-packages  
export PATH=$PATH:/opt/site-packages/local/bin
``` 

#### 运行

##### 基于 ONNX Runtime 运行  
可在开发板或PC运行 

在开发板或PC上，运行以下命令  
```
cd python
python3 infer_onnx.py --img examples/demo01.jpg --model ../models/depth_anything_v2_vits.onnx
```
输出结果
![output](asserts/output-onnx.png)

##### 基于AXEngine运行  
在开发板上运行命令

```
cd python  
python3 infer.py --img examples/demo01.jpg --model ../models/compiled.axmodel 
```  
输出结果
![output](asserts/output-ax.png)


运行参数说明:  
| 参数名称 | 说明  |
| --- | --- | 
| --img | 输入图片路径 | 
| --model | 模型路径 | 

### Latency

#### AX650N

| model | latency(ms) |
|---|---|
|depth_anyting_v2_vits|33.1|


#### AX630C

| model | latency(ms) |
|---|---|
|depth_anyting_v2_vits|300.1|

## 技术讨论

- Github issues
- QQ 群: 139953715
