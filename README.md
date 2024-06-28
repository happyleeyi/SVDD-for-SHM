# SVDD(machine learning) for SHM

## Method

1. prepare dataset (we use 3F shake table test dataset, 8 accelerometers per floor, a total of 24 accelerometers)
2. process dataset (we downsample and cut the training dataset and concatenate 8 acc data per each floor -> (8, 512) for one data)
3. training autoencoder
4. copy the encoder part of the autoencoder and use it as kernel to the hyperspace (we need to place the data in a multi-dimensional space to use svdd)
5. map the training data using the encoder to hyperspace (you can choose the representation dimension of hyperspace as you want)
6. train 3 different svdd models for every 3 floors, using the same kernel function
7. cut the test data for 3 floors ((24, 512) -> (8,512) * 3)
