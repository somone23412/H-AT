H-AT: Hybrid Attention Transfer for Knowledge Distillation
==============

PyTorch training code for:
+ [H-AT: Hybrid Attention Transfer for Knowledge Distillation]()

Origin codes forked from: 
+ https://github.com/szagoruyko/attention-transfer (thanks to Sergey Zagoruyko et al. and their great work)

## Requirements

First install [PyTorch](https://pytorch.org), then install [torchnet](https://github.com/pytorch/tnt):

```
pip install git+https://github.com/pytorch/tnt.git@master
```

then install other Python packages:

```
pip install -r requirements.txt
```

## Experiments
> refer to cifar_train.sh

### train teacher 
```bash
$PATHTOPYTHON/python3 cifar_train.py \
--save logs/resnet_40_1_teacher \
--depth 40 \
--width 1 \
--gpu_id 4
```

### train student
```
TEANET="40_1"
DEPTH=16
WIDTH=1
ALPHA=0 # 0.9
BETA=0 # 1000
GAMMA=0 # 3000
DELTA=10  # 10, delta[] mentioned in paper equals to [0.01, 0.1, 1] * DELTA
EPOCHSTEP="[60,120,160]"
GPUID=1
METHOD=CSHAT

STUNET="$DEPTH"_"$WIDTH"
LOGPATH="$METHOD"_"$STUNET"_"$TEANET"
TEACHER="resnet"_"$TEANET"_"teacher"
TRAIN_FILE="cifar_train.py"

for i in seq 1 5`
do
$PATHTOPYTHON/python3 $TRAIN_FILE \
--save logs/$LOGPATH#$i \
--teacher_id $TEACHER \
--epoch_step $EPOCHSTEP \
--depth $DEPTH \
--width $WIDTH \
--alpha $ALPHA \
--beta $BETA \
--gamma $GAMMA \
--delta $DELTA \
--gpu_id $GPUID
done
```