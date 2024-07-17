# Lets start to contribution!!

## Clone repository

Here´s a simple batch script to clone the repository

````bat
cd C:/Users/User/Desktop
git clone https://github.com/code1O/ai_automat.git
cd C:/Users/User/Desktop/ai_automat
mkdir data
cd data
touch machle_demo_data.csv

````

Insert the next data in `machle_demo_data.csv`

````csv
Car,Model,Volume,Weight,CO2
Toyota,Aigo,182,784,124
Mercedez Benz,Toronto,167,732,154
Lamborghini,Toreador,185,762,175
````

You can use this information for test of machine learning at `Tests/machine_learning.py`

> [!CAUTION]
> For questions of security, the only data file provided is the testing files

## Pip install

There are many modules for install if you wanna contribute to this project

### Prebuilted AI and machine learning

````bash
cd Requirements

python3 -m pip install -r AI.txt

python3 -m pip install -r machle.txt
````

### System handling and web handling

````bash
cd Requirements

python3 -m pip install hand_sys.txt

python3 -m pip install hand_web.txt
````