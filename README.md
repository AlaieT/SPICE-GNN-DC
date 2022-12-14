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

`The goal of the project is to create a model that can significantly reduce simulation time with minimal loss in the accuracy of the numbers.`

<p align='center'>
  <image src='https://github.com/AlaieT/SPICE-GNN/blob/main/picts/power_grid.jpg' width="400"/>
  <p align='center'>Power grid example.</p>
</p>

## Types of analysis

There is two major types of circuits analysis - `DC(Driect Current)` and `Transient`.  
DC analysis - circuit analysis in `T = 0`, so any capasitors or indcutions are not taken into account.  
Transient analysis - circuit analysis in time range from 0 to T(>0), so capasitors or indcutions are accounted for.

`In this project will be realised only DC type of analysis`.

## Data

All train and valid data generated by `generator.py` script.

### Dataset structure

Circuits are described using `SPICE` formate.

- Circuits represent grid of resistors.
- Circtuis may have more then 1 layer
- Circuits have only one voltage source and it is conneced to highest layer
- All layers conneced between by via
- The lower layer the bigger resistance
- Via - resistor with 0 resistance value
- To all nodes in lowermost layer connected current sources

### Data normalization

It is very important to normalize data.  
Every neural network model works better with data scaled to some range.  
I choosed this range of data scaling (-0.5, 0.5).  

ground truth value also should be sacled in some ranage. 
The best variant is just devide by voltage source value to normalize to (0, 1) range.

### Generation

All circuits elements(resistors, voltage and current sources, junctions) considered as graph nodes.

Each circuit divides into:
  - x - maxtrix of shape MxF, M - number of circuit elements and F - size of feature vector.
  - edge_index - descriptions of connections between nodes, from source to target and from target to source
  - mask - allows to select only voltage nodes from model predictions
  - target - target normalized value
  - max_voltage - value of circuit voltage source

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
  - Start of Extractor
  - Save current state in `h`
  - Use conv + relu + conv on `x`
  - Use skipp connection `x=h+x`
  - Linear layer(increases hidden channels size) + relu
  - End of Extractor
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
The point of this function is to show the error in the accuracy of numbers up to a specific number of digits after the decimal point.
The lower metric score the better result.

Example: Acc@1(5.567 and 5.423) = 1, Acc@2(3.2543 and 3.2521) = 0.

![](https://github.com/AlaieT/SPICE-GNN/blob/main/picts/metric.jpg)

Where k - number presission, x - predicted value, y - ground truth value.  
If `(x - y)` rounded to k presission is larger then 0 its return 1 otherwise 0.

## Train

Dataset | Voltage, mV | Resitance, mOm | Cells | Count
--- | --- | --- | --- |--- 
Train dataset | 500-1000  mV | 4 - 100  | 100 - 500 mOm | 27301  
Fold0 dataset | 500-1000  mV | 4 - 100  | 100 - 500 mOm | 3151  
Fold1 dataset | 1000-1500 mV | 4 - 100  | 100 - 500 mOm | 3149  
Fold2 dataset | 1000-1000 mV | 110 - 216 | 100 - 500 mOm | 2015  
Fold3 dataset | 1000-1500 mV | 110 - 216 | 100 - 500 mOm | 2017  

P.S. Train dataset generated with different range step then folds. 

Epochs: 100  
Hidden channels size: 48
Num layers: 3
Optimizer: AdamW(lr=1e-4, weight_decay=1e-5)  
Scheduler: ExponentialLR(gamma=1-1e-6)  
Criterion: L2Error

## Inference

`Fold0`: All values in range of train dataset  
`Fold1`: Voltage values out of range of train dataset  
`Fold2`: Cells number out of range of train dataset  
`Fold3`: Voltage values and  Cells numbers out of range of train dataset  

Metric | Fold0 | Fold1 | Fold2 | Fold3
--- | --- | --- | --- |--- 
Acc@1  | 0% | 0% | 0.377% | 0.31723%
Acc@2  | 11.884% | 11.611% | 61.999% | 62.399%
MAPE  | 0.35701% | 0.36849% | 1.5052% | 1.5032%

Lest remeber goal of this porject - `The goal of the project is to create a model that can significantly reduce simulation time with minimal loss in the accuracy of the numbers.`  

Acc@1 - this metric means that the model can predict values to at least 1 digit after the decimal point, the lower the better.  
Acc@2 - this metric means that the model can predict values to at least 2 digit after the decimal point, the lower the better.  
MAPE - this metric means that the model can predict values with some percentage accurasy, the lower the better.  

We have 4 fold to test the capabilities of created model.  

Looking at the results of the validation we can make several conclusions about the resulting model:
 - The error is small in the area of training values  
 - Increasing the voltage scale has almost no effect on the error value  
 - Increasing the size of circuits greatly affects the accuracy of calculations  
 
Average circuit simulation time using standard mathematical methods is on average equal to 1h:30m(approximate time).  
The average simulation time of circuits when using the resulting model is on average equal to 0h:3m:42s(approximate time).

Let's do some simple calculations:

Tm - simulation time using standard mathematical methods.  
Tnn - simulation time using nn model.  
Td = Tm/Tnn ~= 22 times.  

AvgAcc@1 = 0.174%  
AvgAcc@2 = 36.97%  
AvgMAPE = 0.933%  

Let's choose AvgMAPE as well as the ratio of time Td.  
So using model, we can reduce the simulation time by a factor of 22, but we will get an error of 0.933% of MAPE in average.
It turns out a significant reduction in time for a small error.  

This model can perform better with a larger data set and better training parameters.  
But even now we can say that the use of neural networks to reduce the development and testing time of integrated circuits can significantly increase the productivity of the development of new devices.


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

## Scripts

List of available scripts:

- src/generator.py - allows to generate dataset.  
  -m - name or type of dataset  
  -vr - voltage range in mV (min, max, step)  
  -rr - resistance range in mOm (min, max, step)  
  -cr - cells range (min, max, step)  
  -nl - numver of layers  

  Usage example - `python ./src/generator.py -m fold0_big -vr 510 1020 60 -cr 4 20 3 -rr 110 520 60 -nl 2`

- src/train.py - allows to train model.  
  -ft - csv file that contains train data  
  -fv - csv files that contains validation data  
  -e epochs count  
  -bt - train batch size  
  -bv - validation batch size  

  Usage example - `python ./src/train.py -ft ./assets/train_exp.csv -fv ./assets/fold0_exp.csv ./assets/fold1_exp.csv -e 1000 -bt 32 -bv 64`
  
- src/analysis.py - allows to analysis data: statistics of generated data, heatmap of circtui ground truth and plot loss surface of model.  
  -gd - analysis of generated data, max-min-mean voltage dropout in datases  
  -ls - plot loss surface of model  
  -p - path to valid or trian csv file  
  -m - path to model checkpoint file  

  Usage examples:  
   `python ./src/analysis.py -gd -p ./assets/train.csv`,  
   `python ./src/analysis.py -ls -p ./assets/train.csv`
   
## Tasks

- Find best set of parameters for model
- Add transient dataset structure
- Method for transient analysis based on DC GNN model
