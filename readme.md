# Dueling Double Deep Q Learning Network for Atari Breakout
This repo provides code to train an agent to play Atari Breakout game using Dueling Double Deep Q Learning Network, a type of Reinforcement Learning.

In my experiments the model was trained on NVIDIA GTX 1050Ti Mobile GPU for 7200 episodes. The model was able to achieve a score of maximum score of 340.


## Dependencies
- pip install tensorflow
- pip install gym==0.26.2
- pip install cv2

Note: Make sure the gym version has `BreakoutDeterministic-v4`.

## Training the model

To train the model, with the hyper-parameters used in the code, run the cells or run the `breakout_DDDQN.py` file. The model will be saved in the `model_saves` folder.

The model hyper-parameters can be changed to experiment with different configurations of the model and also based on the hardware available.

## Evaluating the model

To evaluate the model, run the cells under `Evaluation of the model` markdown cell. The model will be loaded from the `model_saves` folder and the model will play the game for the number of episodes in declared in `eval_length` hyper-parameter.

This will also save the frames of the game played by the model in a gif file format, named `animation.gif` in the root directory.

## Sample gameplay

![Sample Atari Breakout gameplay by Reinforcement Learning model - Tejus Vignesh Vijayakumar](https://github.com/tejus-vignesh/Dueling-Double-Deep-Q-Learning-Network-for-Atari-Breakout/blob/main/animation.gif?raw=true)
