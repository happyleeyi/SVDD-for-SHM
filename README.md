# SVDD(machine learning) for SHM

## Introduction
![Multi Deep svdd model pic - 복사본](https://github.com/happyleeyi/SVDD-for-SHM/assets/173021832/9f614f57-9359-4d56-bd22-31bef9b7de67)

The research to suggest a method to find whether and where the defect happens in the building on real-time.

We use an autoencoder to map the acceleration data to hyperspace and the SVDD method(machine learning) to find the location of the defect in this section.

## Dataset

3-story shake table test data with an undamaged case and several damaged cases

![image](https://github.com/happyleeyi/SVDD-for-SHM/assets/173021832/b328aac8-2b3d-4063-9154-6c3e3cb029e1)


(open-source by Engineering Institute at Los Alamos National Laboratory)

## Method

### 1. data processing
1. prepare dataset (we use 3F shake table test dataset, 8 accelerometers per floor, a total of 24 accelerometers)
2. process dataset (we downsample and cut the training dataset and concatenate 8 acc data per each floor -> training dataset : (8, 512) for one data)

### 2. Training
![Multi svdd model pic](https://github.com/happyleeyi/SVDD-for-SHM/assets/173021832/255e7a82-2bc4-4ad6-b165-6b4a0d51e6d6)

1. training autoencoder
2. copy the encoder part of the autoencoder and use it as kernel to the hyperspace (we need to place the data in a multi-dimensional space to use svdd)
3. map the training data using the encoder to hyperspace (you can choose the representation dimension of hyperspace as you want)
4. train 3 different svdd models for every 3 floors, using the same kernel function

### 3. Test
1. cut the test data for 3 floors (test dataset : (24, 512) -> (8,512) * 3 for one data)
2. put the data into the encoder and map to hyperspace for every floor. (total 3 datas per one data in this dataset)

![image](https://github.com/happyleeyi/SVDD-for-SHM/assets/173021832/9113a39a-d7fb-4f44-84e5-5245d848daaa)

3. if every data is mapped into the sphere of the corresponding floor's svdd model, the building is on normal state.
4. however, if several datas are mapped outside of the sphere of the corresponding floor's svdd model, the floor that the data mapped the furthest from the corresponding sphere is determined to be damaged.

## Result

![image](https://github.com/happyleeyi/SVDD-for-SHM/assets/173021832/cbc72996-60e9-44bc-8673-531957db65d8)
![image](https://github.com/happyleeyi/SVDD-for-SHM/assets/173021832/73e82b5c-6e0a-48e4-89d5-6ba4b338b568)
![image](https://github.com/happyleeyi/SVDD-for-SHM/assets/173021832/258888e5-440a-4e28-8124-fbb01f722b06)




