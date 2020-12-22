# ml-tech-final-project
This is the final project of the course "CSIE 5433 Machine Learning Techniques" at NTU 2020 fall.

Dataset: [Hotel booking demands](https://www.kaggle.com/jessemostipak/hotel-booking-demand)

## Dependencies
- python
- argparse
- numpy
- pandas

I tested the code on my computer with the env. of `python=3.6.12`  and `torch=1.1.0`. I have not coded the gpu part for torch.

## To ChangLee
Belows are the skeleton of our training module:
- `options.py`: class for managing experiment arguments (usages are in the file)
- `dataloader.py`: class for loading data (usages are in the file)
- `model/`: you can wrote your model file and put it in `model/`. For the format and usage, please see the template `model_template.py` and a small example `cancel_toy_model.py` I wrote for reference.
- `util.py`: some small functions

## Quick Start
I wrote a small example of **a cancellation model**, a 1 hidden-layer perceptron that predicts whether a booking request will be cancelled or not. To see how it work, please run
```
python train.py --model cancel
```

Still some bugs due to nan in the dataset. Need some data pre-processing. I will fix it later.



