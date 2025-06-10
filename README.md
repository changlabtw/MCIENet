# MCIENet
MCIENet: Multi-scale CNN-based Information Extraction from DNA Sequences for 3D chromatin interactions Prediction

![](figures/fig1-a_Workflow.png)

# get started
## setup environment
### use docker
建立容器，並進入容器
```shell
# 構建並啟動容器（後台運行）
docker-compose -f docker/docker-compose.yml up -d

# 進入容器
docker-compose -f docker/docker-compose.yml exec mcienet /bin/bash
```
你可以開始在 command line 使用 MCIENet 了

退出容器
```shell
exit
```

停止並刪除容器
```shell
docker-compose -f docker/docker-compose.yml down
```

### use scripts
```shell
call "scripts\env\Win\set-env_conda.bat" # conda
call "scripts\env\Win\set-env_venv.bat" # venv
```
## generate train data

## Reference
- _Cao, Fan, et al. "Chromatin interaction neural network (ChINN): a machine learning-based method for predicting chromatin interactions from DNA sequences." Genome biology 22 (2021): 1-25. https://doi.org/10.1186/s13059-021-02453-5._
  - Github: https://github.com/mjflab/chinn
- _Zhou, Zhihan, et al. "Dnabert-2: Efficient foundation model and benchmark for multi-species genome." arXiv preprint arXiv:2306.15006 (2023). https://doi.org/10.48550/arXiv.2306.15006._
  - Github: https://github.com/MAGICS-LAB/DNABERT_2
  - Pretrain model: https://huggingface.co/zhihan1996/DNABERT-2-117M

## Citation

This version of implementation is only for learning purpose. For research, please refer to  and  cite from the following paper:
```
@inproceedings{ MCIENet,
  author = "Yen-Nan Ho and Jia-Ming Chang"
  title = "MCIENet: Multi-scale CNN-based Information Extraction from DNA Sequences for 3D chromatin interactions Prediction",
  booktitle = "",
  pages = "",
  year = "2025",
}
```
