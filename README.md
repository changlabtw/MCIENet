# MCIENet
MCIENet: Multi-scale CNN-based Information Extraction from DNA Sequences for 3D chromatin interactions Prediction

![](figures/fig1-a_Workflow.png)

## setup environment
### docker
```shell
# 構建並啟動容器（後台運行）
docker-compose up -d

# 進入容器
docker-compose exec mcienet /bin/bash
```

#### 停止
```shell
docker-compose down
```

### scripts
```shell
call "scripts\env\Win\set-env_conda.bat" # conda
call "scripts\env\Win\set-env_venv.bat" # venv
```


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
