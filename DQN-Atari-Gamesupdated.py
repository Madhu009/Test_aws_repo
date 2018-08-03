

# coding: utf-8

# In[1]:

import gym
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from collections import deque
import random

import warnings
warnings.filterwarnings('ignore')

env=gym.make('BreakoutDeterministic-v4')
state = env.reset()


actions=env.unwrapped.get_action_meanings()
nA=len(actions)-1
ATARI_SHAPE = (84, 84, 4)
ACTION_SIZE = 3

# In[5]:

#preprocessing the image

def pre_process(frame_array):
    
    #converting into graysclae since colors don't matter
    
    from skimage.color import rgb2gray
    grayscale_frame = rgb2gray(frame_array)
    
    # resizing the image 
    from skimage.transform import resize
    resized_frame = np.uint8(resize(grayscale_frame, (84, 84), mode='constant') * 255)
    
#     return np.reshape([resized_frame], (1, 84, 84, 1))
    
    return resized_frame


# In[6]:




# In[7]:

def epslion_greedy_policy_action(current_state,episode):
	Q_value = model.predict([current_state,np.ones(ACTION_SIZE).reshape(1, ACTION_SIZE)])
	return np.argmax(Q_value[0])


# In[8]:

from keras import backend as K

def huber_loss(y, q_value):
    error = K.abs(y - q_value)
    quadratic_part = K.clip(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
    return loss


# In[9]:

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import RMSprop

# deep mind model 
def deepmind_model_atari():
    
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',input_shape=(84, 84, 4)))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nA))
    model.summary()
    optimizer = RMSprop(lr=0.0039, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss=huber_loss)
    return model


from keras import layers
from keras.models import Model,load_model
def atari_model():
    # With the functional API we need to define the inputs.
    frames_input = layers.Input(ATARI_SHAPE, name='frames')
    actions_input = layers.Input((ACTION_SIZE,), name='action_mask')

    # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
    normalized = layers.Lambda(lambda x: x / 255.0, name='normalization')(frames_input)

    # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
    conv_1 = layers.convolutional.Conv2D(
        16, (8, 8), strides=(4, 4), activation='relu'
    )(normalized)
    # "The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
    conv_2 = layers.convolutional.Conv2D(
        32, (4, 4), strides=(2, 2), activation='relu'
    )(conv_1)
    # Flattening the second convolutional layer.
    conv_flattened = layers.core.Flatten()(conv_2)
    # "The final hidden layer is fully-connected and consists of 256 rectifier units."
    hidden = layers.Dense(256, activation='relu')(conv_flattened)
    # "The output layer is a fully-connected linear layer with a single output for each valid action."
    output = layers.Dense(ACTION_SIZE)(hidden)
    # Finally, we multiply the output by the mask!
    filtered_output = layers.Multiply(name='QValue')([output, actions_input])

    model = Model(inputs=[frames_input, actions_input], outputs=filtered_output)
    model.summary()
    optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    # model.compile(optimizer, loss='mse')
    # to changed model weights more slowly, uses MSE for low values and MAE(Mean Absolute Error) for large values
    model.compile(optimizer, loss=huber_loss)
    return model
    
replay_memory = deque(maxlen=400000)
model = load_model("atari_model25900.h5",custom_objects={'huber_loss': huber_loss})
target_model = load_model("atari_model25900.h5",custom_objects={'huber_loss': huber_loss})

nEpisodes = 50000
total_observe_count = 2
epslion = 0.1
batch_size = 32
gamma = 0.99
final_epsilon = 0.1
epsilon_step_num = 100000
epsilon_decay = (1.0 - final_epsilon) / epsilon_step_num
max_score = 0
target_model_change = 5

# In[10]:



# In[11]:

def get_sample_random_batch_from_replay_memory():
    
    mini_batch = random.sample(replay_memory,batch_size)
    
    current_state_batch = np.zeros((batch_size, 84,84, 4))
    next_state_batch = np.zeros((batch_size, 84,84, 4))
   
    
    actions, rewards, dead = [], [], []
    
    for idx, val in enumerate(mini_batch):
        
        current_state_batch[idx] = val[0]
        actions.append(val[1])
        rewards.append(val[2])
        next_state_batch[idx] = val[3]
        dead.append(val[4])
    
    return current_state_batch , actions, rewards, next_state_batch, dead


def deepQlearn():
    
    current_state_batch , actions, rewards, next_state_batch, dead = get_sample_random_batch_from_replay_memory()
    
    actions_mask = np.ones((batch_size,ACTION_SIZE))
    next_Q_values = target_model.predict([next_state_batch,actions_mask])  # separate old model to predict
    
    targets = np.zeros((batch_size,))
    
    for i in range(batch_size):
        if dead[i]:
            targets[i] = -1
        else:
            targets[i] = rewards[i] + gamma * np.amax(next_Q_values[i])
            
    one_hot_actions=np.eye(ACTION_SIZE)[np.array(actions).reshape(-1)]
    one_hot_targets = one_hot_actions * targets[:, None]
    
    model.fit([current_state_batch,one_hot_actions], one_hot_targets, epochs=1,batch_size=batch_size, verbose=0)
    


def save_model(episode):
    model_name = "atari_model{}.h5".format(episode)
    model.save(model_name)


# In[ ]:

def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


for episode in range(nEpisodes):
    
    dead, done, lives_remaining,score = False, False, 5, 0
    
    current_state = env.reset()
    for _ in range(random.randint(1, 30)):
        current_state, _, _, _ = env.step(1)
        
    current_state = pre_process(current_state)
    current_state = np.stack((current_state, current_state, current_state, current_state), axis=2)
    current_state = np.reshape([current_state], (1, 84, 84, 4))
    
    while not done:
    
        action = epslion_greedy_policy_action(current_state,episode)
        real_action = action + 1
        
        
        next_state, reward, done, lives_left = env.step(real_action)
        
        next_state = pre_process(next_state) # 84,84 grascale frame 
        next_state = np.reshape([next_state], (1, 84, 84, 1))
        next_state = np.append(next_state, current_state[:, :, :, :3], axis=3)
        
        if lives_remaining > lives_left['ale.lives']:
            dead = True
            lives_remaining = lives_left['ale.lives']

        
        replay_memory.append((current_state, action, reward, next_state, dead))
        
        if episode>total_observe_count:
            deepQlearn()
            
            if episode % target_model_change == 0:
                target_model.set_weights(model.get_weights())
      
            
        score += reward
        
        if dead:
            dead = False
        else:
            current_state = next_state
            
        if max_score<score:
            print("max score for the episode {} is : {} ".format(episode,score))
            max_score = score
            #save_model(episode)

    if episode%1 == 0:
        print("final score for the episode {} is : {} ".format(episode,score))
        #save_model(episode)
    

