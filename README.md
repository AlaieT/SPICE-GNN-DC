# SPICE-GNN

Graph Neural Networks for modeling circuits described using SPICE

## Porpuse of the project

Well known fact that neural networks can make your life much easer.
One of the main advantages of neural netowrks is speed of matrix calculation using cgu.

In [integrated circuit](https://en.wikipedia.org/wiki/Integrated_circuit) we have a lot of elements(VLSI - 20 000 to 1 000 000, ULSI - 1 000 000 and more).  
That means if we whant to model such curcuits we need a lot of time(it can long days).

So the main porpuse of this poject it is decreas modeling time with help of neural networks.

## Project limitations

This will be too hard to start straight with modeling integrated circuit that have `transistors`.  
So for the start of this project there is no transisors in circuits.

If there is no transistor then there is no standart cell that describes logical functions, triggers and e.t.c.  
So what is left it's - `power grid` of integrated circuit.

<p align='center'>
<image src='https://github.com/AlaieT/SPICE-GNN/blob/main/picts/power_grid.png'/>
<p align='center'>Example of power grid</p>
<p>

## Types of analysis

Due to project limitations only circuit `power grid` will be modeled.

There is two major types of circuits analysis - `DC(Driect Current)` and `Transient`.  
DC analysis - curctuit analysis in time `T = 0`, so any capasitors or indcutions are not taken into account.  
Transient analysis - curctuit analysis in time range from 0 to T, so capasitors or indcutions are accounted for.

In DC and Transient analysis will be recived voltage values in each node - So Called IRDrop.  
In this project will be realised onlyt `DC` type of analysis.

## Dataset structure

Circuits are described using `SPICE`.

- Circuits represent grid of resistors, grids are squares.
- Circtuis may have more then 1 layer
- Circuits have only one voltage source it is conneced to highest layer
- All layers conneced between by via
- All resistors in one layer have the same resictance value
- The lower layer the bigger resistance
- Via - resistor with 0 resistance value
- To all nodes in lowermost layer connected current sources
- There is no random values in dataset

## Data normalization

It is very important to normalize data.  
Every neural network model works better with data scaled to range (0, 1).  
For us this is very important, because we whant use our model to predict values from different circuits with different ranges of values.

We also have to scale ground truth value.  
The best variant is just devide by voltage source value to normalize every circuit ground truth to range (0, 1).

## Neural Network Model

The first thing that comes to mind its CNN architecture, but such model have some limitaions like: size of item and data complexety.  
As mentioned early circuits can reach in size 1M+ elements, so imagen how large will be attention matrix of such circuits.  
Also circuit type of data is too hard represent in matrix form.

The proper way to represent circuit data - graph. Such representation can handle any complexety circuit. It also require much less memory then annotation matrix.

[GNN(Graph Neural Network)](https://en.wikipedia.org/wiki/Graph_neural_network) - can preform such tasks as: node classifiaction, node regeression, connection prediction, graph classification and e.t.c.

## Traning

Training approach of GNN model is not so different from other neural networks.  
Method of learning - [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning).

## Loss and Metric

### Loss

We have reggression task, so loss function can be - `L1Loss`, `MSELoss`, `HuberLoss`. This is standar loss functions in every library.
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

## Illustrations of results

### Train loss

<img src='https://github.com/AlaieT/SPICE-GNN/blob/main/picts/loss.png' alt='drawing' width='500'/>

### Acc@1 metrick

<img src='https://github.com/AlaieT/SPICE-GNN/blob/main/picts/acc1.png' alt='drawing' width="500"/>

### Acc@2 metrik

<img src='https://github.com/AlaieT/SPICE-GNN/blob/main/picts/acc2.png' alt='drawing' width="500"/>

### L2Error loss surface

<img src='https://github.com/AlaieT/SPICE-GNN/blob/main/picts/deep_nn_loss_log_surface.gif' alt='drawing' width='500'/>
                   
## Tasks

- Find best set of parameters for model
- Add transient dataset structure
- Method for transient analysis based on DC GNN model
