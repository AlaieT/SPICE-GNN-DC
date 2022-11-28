# SPICE-GNN

Graph Neural Networks for modeling circuits described using SPICE

## About project

Today neural networks are using in many aspects of our lifes.  
One of the main advantages of neural networks that they are universal function approximator.

In [integrated circuit](https://en.wikipedia.org/wiki/Integrated_circuit) we have a lot of elements(VLSI - 20 000 to 1 000 000, ULSI - 1 000 000 and more).  
That means if we whant to model such curcuits to hight number precision we need a lot of time(it can long days).

The main porpuse of this poject it is decreas modeling time with help of neural networks.

## Project purpose

This will be too hard to start straight with modeling integrated circuit that have `transistors`.    
To simplify this project we wil be using circtuits that doesn't have transistors.  

IC usialy splits by two parts: power grid and transistors on cristal.Power gird is used for transfer voltage to all transistors on cristal.The main porblem of power grid - voltage drop out can be too hight in some places of grid. So the main purpose of modeling power grid it is find all area where voltage drop out is higher than some threshold value.  

As described early to achieve hight value accurasy using standard math methods we need a lot of time.
Example: to achieve numbers accurate to two decimal places in PG with ~10k nodes needs at least 4h of modeling time.

<p align='center'>
  <image src='https://github.com/AlaieT/SPICE-GNN/blob/main/picts/power_grid.jpg' width="400"/>
  <p align='center'>Power grid example.</p>
</p>

## Types of analysis

There is two major types of circuits analysis - `DC(Driect Current)` and `Transient`.  
DC analysis - circuit analysis in `T = 0`, so any capasitors or indcutions are not taken into account.  
Transient analysis - circuit analysis in time range from 0 to T(>0), so capasitors or indcutions are accounted for.

`In this project will be realised only DC type of analysis`.

## Dataset structure

Circuits are described using `SPICE` formate.

- Circuits represent grid of resistors.
- Circtuis may have more then 1 layer
- Circuits have only one voltage source and it is conneced to highest layer
- All layers conneced between by via
- The lower layer the bigger resistance
- Via - resistor with 0 resistance value
- To all nodes in lowermost layer connected current sources

## Data normalization

It is very important to normalize data.  
Every neural network model works better with data scaled to some range.  
I choosed this range of data scaling (-0.5, 0.5).  

ground truth value also should be sacled in some ranage. 
The best variant is just devide by voltage source value to normalize to (0, 1) range.

## Neural Network Model

The first thing that comes to mind its CNN architecture, but such model have some limitaions like: size of item and data complexety.  
As mentioned early circuits can reach in size 1M+ elements, so imagen how large will be adjacency matrix of such circuits.  
Also circuits is too hard represent in matrix form withot limitations.

The proper way to represent circuit data - graph. Such representation can handle any complexety circuit and have no limitations. 

[GNN(Graph Neural Network)](https://en.wikipedia.org/wiki/Graph_neural_network) - can preform such tasks as: node classifiaction, node regeression, connection prediction, graph classification and e.t.c.

<p align='center'>
  <img src='https://github.com/AlaieT/SPICE-GNN/blob/main/picts/model.jpg' alt='drawing' width='300'/>
  <p align='center'>Model architecture.</p>
</p>

### Model discription
  - MLP layer encoding input features of `x`
  - Save current state in `h`
  - Use conv + relu + conv on `x`
  - Use skipp connection `x=h+x`
  - With linear layer expand number of features of `x` + relu
  - MPL layer decoding output features of `x` in needed shape
  
## Loss and Metric

### Loss

We have reggression task, we can choose one of standart loss functions - `L1Loss`, `MSELoss`, `HuberLoss`. 
But in order to have hight accurasy values we need more proper regression loss function.

For this we need such type of loss function, that can better focuse on small values.
I made function that can preform better than any of standart regression ones, its called `L2Error`.

![](https://github.com/AlaieT/SPICE-GNN/blob/main/picts/loss.jpg)

Where x - predicted values, y - ground truth values and p - model parameters(aka neurons)

### Metric

We whant to recive hight accurasy values from model, then we have properly validate it.  
The best wariant is use so called `Acc@k` or top k accurasy metric.  
This metric used in multy label calssification tasks, but i rebuilded it to our porpuses.

![](https://github.com/AlaieT/SPICE-GNN/blob/main/picts/metric.jpg)

Where k - number presission, x - predicted value, y - ground truth value.  
If `(x - y)` rounded to k presission is larger then 0 its return 1 otherwise 0.

## Scripts

List of available scripts:

- src/generator.py - allows to generate dataset.  
  Usage example - `python ./src/generator.py -m train -vr 500 1020 20 -cr 1 11 1 -rr 100 520 20 -nl 4`
- src/train.py - allows to train model.  
  Usage example - `python ./src/train.py -f ./assets/train.csv -e 500 -bt 128 -bv 128 -r`
- src/test.py - allows to run test on several folds.  
  Usage example - `python ./src/test.py -f ./assets/test_fold1.csv ./assets/test_fold2.csv -mp ./dict/dnn/best.pt -bs 256 -r`
- src/analysis.py - allows to analysis data: statistics of generated data, heatmap of circtui ground truth and plot loss surface of model.  
  Usage examples:  
   `python ./src/analysis.py -gd -p ./assets/train.csv`,  
   `python ./src/analysis.py -ch -p ./assets/train/ibmpg0/ibmpg0.csv`,  
   `python ./src/analysis.py -ls -p ./assets/train.csv`

## Results

Epochs: 100  
Optimizer: AdamW(lr=1e-4, weight_decay=1e-5)  
Scheduler: ExponentialLR(optimizer, gamma=1-1e-6)  
Criterion: L2Error

`Fold0`: All values in range of train dataset  
`Fold1`: Voltage values out of range of train dataset  
`Fold2`: Cells number out of range of train dataset  
`Fold3`: Voltage values and  Cells numbers out of range of train dataset  

Metric | Fold0 | Fold1 | Fold2 | Fold3
--- | --- | --- | --- |--- 
Acc@1  | 0% | 0% | 0.377% | 0.31723%
Acc@2  | 11.884% | 11.611% | 61.999% | 62.399%
MAPE  | 0.35701% | 0.36849% | 1.5052% | 1.5032%

### MAPE, Acc@1 metrick, Acc@2 metrik

<p float="left">
  <img src='https://github.com/AlaieT/SPICE-GNN/blob/main/picts/mape.png' alt='drawing' width='250'/>
  <img src='https://github.com/AlaieT/SPICE-GNN/blob/main/picts/acc1.png' alt='drawing' width="250"/>
  <img src='https://github.com/AlaieT/SPICE-GNN/blob/main/picts/acc2.png' alt='drawing' width="250"/>
</p>

### L2Error contour and loss surface

<p float="left">
  <img src='https://github.com/AlaieT/SPICE-GNN/blob/main/picts/deep_nn_loss_log_surface.gif' alt='drawing' width='350'/>
  <img src='https://github.com/AlaieT/SPICE-GNN/blob/main/picts/deep_nn_loss_log_contour.png' alt='drawing' width='350'/>
</p>

## Tasks

- Find best set of parameters for model
- Add transient dataset structure
- Method for transient analysis based on DC GNN model