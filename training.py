import numpy as np
import gym
from gym import spaces
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
from collections import deque

# Constants
NUM_AGVS = 10
MAX_BATTERY = 100
CHARGING_RATE = 5
DISCHARGE_RATE = 1
MAX_TASKS = 20

class AGVEnv(gym.Env):
    def __init__(self):
        super(AGVEnv, self).__init__()
        self.action_space = spaces.MultiDiscrete([MAX_TASKS + 2] * NUM_AGVS)
        self.observation_space = spaces.Box(
            low=0, 
            high=100, 
            shape=(NUM_AGVS * 4 + MAX_TASKS * 3 + 1,), 
            dtype=np.float32
        )
        self.reset()
    
    def reset(self):
        self.agvs = [{'x': random.randint(0, 99), 'y': random.randint(0, 99), 'battery': MAX_BATTERY, 'status': 0} for _ in range(NUM_AGVS)]
        self.tasks = [{'x': random.randint(0, 99), 'y': random.randint(0, 99), 'priority': random.randint(1, 5)} 
                      for _ in range(random.randint(1, MAX_TASKS))]
        self.demand = len(self.tasks)
        self.time_step = 0
        return self._get_obs()
    
    def step(self, action):
        reward = 0

        for agv_id, task_action in enumerate(action):
            agv = self.agvs[agv_id]
            
            if task_action == 0:  # Park
                if agv['status'] != 0:
                    agv['status'] = 0
                    reward += 0.1  # No reward for parking when not needed
            elif task_action == 1:  # Charge
                if agv['battery'] < MAX_BATTERY * 0.5:
                    agv['status'] = 1
                    agv['battery'] = min(agv['battery'] + CHARGING_RATE, MAX_BATTERY)
                    reward += 2 * (1 - agv['battery'] / MAX_BATTERY)
                else:
                    reward -= 1  # Penalty for unnecessary charging
            elif task_action - 2 < len(self.tasks):  # Assign to task
                task = self.tasks[task_action - 2]
                distance = np.sqrt((agv['x'] - task['x'])**2 + (agv['y'] - task['y'])**2)
                if agv['battery'] >= distance * DISCHARGE_RATE:
                    agv['x'], agv['y'] = task['x'], task['y']
                    agv['battery'] -= DISCHARGE_RATE * distance
                    reward += (task['priority'] * 3) - (distance / 200)  # Reward based on priority and distance
                    self.tasks.pop(task_action - 2)
                    agv['status'] = 2
                else:
                    reward -= 1  # Penalty for attempting task with insufficient battery
            
            if agv['battery'] < 10:
                reward -= (20 - agv['battery']) / 20  # Penalty for low battery
        
        self.demand = len(self.tasks)
        self.time_step += 1
        
        if random.random() < 0.3:
            self.tasks.append({'x': random.randint(0, 99), 'y': random.randint(0, 99), 'priority': random.randint(1, 5)})
        
        done = self.time_step >= 100 or len(self.tasks) == 0 or all(agv['battery'] <= 0 for agv in self.agvs)
        
        reward = reward / 100
        
        return self._get_obs(), reward, done, {}
    
    def _get_obs(self):
        obs = []
        for agv in self.agvs:
            obs.extend([agv['x'] / 100, agv['y'] / 100, agv['battery'] / MAX_BATTERY, agv['status'] / 2])
        for task in self.tasks[:MAX_TASKS]:
            obs.extend([task['x'] / 100, task['y'] / 100, task['priority'] / 5])
        obs.extend([0, 0, 0] * (MAX_TASKS - len(self.tasks)))
        obs.append(self.demand / MAX_TASKS)
        return np.array(obs, dtype=np.float32)

def create_q_model(action_dim, obs_shape):
    inputs = layers.Input(shape=obs_shape)
    layer1 = layers.Dense(256, activation="relu")(inputs)
    layer2 = layers.Dense(128, activation="relu")(layer1)
    
    outputs = [layers.Dense(action_dim, activation="linear", name=f"agv_{i}_output")(layer2) for i in range(NUM_AGVS)]
    
    return keras.Model(inputs=inputs, outputs=outputs)

class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.priorities = []
        self.capacity = capacity
    
    def add(self, experience):
#         print("Debug: Adding experience to buffer")
#         print(f"Debug: Current buffer length: {len(self.buffer)}")
#         print(f"Debug: Current priorities length: {len(self.priorities)}")
#         print(f"Debug: Type of self.priorities: {type(self.priorities)}")
#         if self.priorities:
#             print(f"Debug: First few priorities: {self.priorities[:5]}")
        
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
            self.priorities.pop(0)
        self.buffer.append(experience)
        
        if self.priorities:
#             print("Debug: About to calculate max_priority")
            max_priority = max(self.priorities)
        else:
            max_priority = 1.0
        
#         print(f"Debug: Calculated max_priority: {max_priority}")
        self.priorities.append(max_priority)
#         print("Debug: Experience added successfully")
    
    def sample(self, batch_size):
        alpha = 0.6
        probs = np.array(self.priorities, dtype=np.float32) ** alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        return samples, indices
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            if isinstance(priority, (np.ndarray, list)):
                # If priority is an array or list, take the mean
                self.priorities[idx] = float(np.mean(priority))
            else:
                # If it's already a scalar, convert to float
                self.priorities[idx] = float(priority)

def compute_n_step_returns(rewards, next_q_values, dones, gamma, n_steps):
    returns = np.zeros_like(rewards)
    batch_size = len(rewards)
    
    # Reshape next_q_values to [batch_size, NUM_AGVS * action_dim]
    next_q_values = tf.reshape(next_q_values, [batch_size, -1])
    
    for t in range(batch_size):
        n_step_return = 0
        for n in range(min(n_steps, batch_size - t)):
            n_step_return += gamma**n * rewards[t+n]
            if t+n == batch_size - 1 or dones[t+n]:
                break
        if t + n < batch_size - 1 and not dones[t+n]:
            next_q = tf.reduce_max(next_q_values[t+n])
            n_step_return += gamma**(n+1) * next_q
        returns[t] = n_step_return
    return returns

batch_size = 128
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.999
learning_rate = 0.0001
n_steps = 2

env = AGVEnv()
action_dim = MAX_TASKS + 2
model = create_q_model(action_dim, env.observation_space.shape)
model_target = create_q_model(action_dim, env.observation_space.shape)


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0001,
    decay_steps=1000,
    decay_rate=0.9)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
# optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

# replay_buffer = PrioritizedReplayBuffer(100000)
replay_buffer = PrioritizedReplayBuffer(capacity=2000000)  

@tf.function
def train_step(model, model_target, optimizer, states, actions, returns, next_states, dones):
    future_rewards = model_target(next_states, training=False)
    
    with tf.GradientTape() as tape:
        q_values = model(states, training=True)
        q_values = tf.reshape(q_values, [tf.shape(states)[0], NUM_AGVS, -1])
        action_masks = [tf.one_hot(actions[:, i], action_dim) for i in range(NUM_AGVS)]
        q_action = tf.stack([tf.reduce_sum(tf.multiply(q_values[:, i], action_masks[i]), axis=-1) for i in range(NUM_AGVS)], axis=-1)
#         loss = tf.reduce_mean(tf.square(tf.expand_dims(returns, -1) - q_action))
        loss = tf.keras.losses.Huber()(tf.expand_dims(returns, -1), q_action)
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# Main training loop
episodes = 3000
for episode in range(episodes):
    state = env.reset()
    episode_reward = 0
    done = False
    episode_loss = []
    
    while not done:
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = model(state_tensor, training=False)
        action_probs = tf.reshape(action_probs, [1, NUM_AGVS, -1])
        
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = [tf.argmax(action_probs[0, i]).numpy() for i in range(NUM_AGVS)]
        
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        
#         print("Debug: About to add experience to replay buffer")
#         print(f"Debug: State shape: {np.shape(state)}")
#         print(f"Debug: Action shape: {np.shape(action)}")
#         print(f"Debug: Reward: {reward}")
#         print(f"Debug: Next state shape: {np.shape(next_state)}")
#         print(f"Debug: Done: {done}")
        
        replay_buffer.add((state, action, reward, next_state, done))
        
        state = next_state
        
        if len(replay_buffer.buffer) >= batch_size:
            batch, indices = replay_buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
            
            states = tf.convert_to_tensor(states, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.int32)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)
            
            next_q_values = model(next_states, training=False)
            returns = compute_n_step_returns(rewards, next_q_values, dones, gamma, n_steps)
            
            loss = train_step(model, model_target, optimizer, states, actions, returns, next_states, dones)
            episode_loss.append(loss.numpy())
            
            # Update priorities
            td_errors = np.abs(returns - tf.reduce_max(model(states, training=False), axis=-1).numpy())
            new_priorities = td_errors + 1e-6  # Small constant to avoid zero priority
            new_priorities = np.mean(new_priorities, axis=-1)  # Take mean across AGVs if necessary
            replay_buffer.update_priorities(indices, new_priorities)
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
    if episode % 5 == 0:
        model_target.set_weights(model.get_weights())
    
    if episode % 10 == 0:
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        print(f"Episode: {episode}, Reward: {episode_reward}, Epsilon: {epsilon}, Avg Loss: {avg_loss}")
    
    if episode % 100 == 0:
        model.save(f'agv_task_assignment_model_episode_{episode}.h5')
    
    if episode >= 220 and episode % 20 == 0:
        model.save(f'agv_task_assignment_model_episode_{episode}.h5')

model.save('agv_task_assignment_model.h5')


def interpret_action(action):
    if action == 0:
        return "Parking"
    elif action == 1:
        return "Charging"
    else:
        return f"Going to task {action - 1}"

def use_model(model, state):
    state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
    state_tensor = tf.expand_dims(state_tensor, 0)
    action_probs = model(state_tensor, training=False)
    actions = [tf.argmax(a[0]).numpy() for a in action_probs]
    
    for i, action in enumerate(actions):
        print(f"AGV {i}: {interpret_action(action)}")

# Uncomment these lines to test the model
# state = env.reset()
# use_model(model, state)

'''
if needed
- update more frequently, isntead of 10 do 5
'''