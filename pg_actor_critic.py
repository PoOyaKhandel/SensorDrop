import random
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


#--------------------------------------


class PolicyGradientActorCritic(object):

  def __init__(self, session,
                     optimizer,
                    #  actor_network,
                    #  critic_network,
                     state_dim,
                     num_actions,
                     init_exp=0.4,         # initial exploration prob
                     final_exp=0.0,        # final exploration prob
                     anneal_steps=1500,    # N steps for annealing exploration
                     discount_factor=0.99, # discount future rewards
                     reg_param=0.001,      # regularization constants
                     max_gradient=5,       # max gradient norms
                     summary_writer=None,
                     summary_every=100,
                     load_model=0,
                     if_train=1):

    # tensorflow machinery
    self.session        = session
    self.optimizer      = optimizer
    self.summary_writer = summary_writer

    # model components
    # self.actor_network  = actor_network
    # self.critic_network = critic_network

    # training parameters
    self.state_dim       = state_dim
    self.num_actions     = num_actions
    self.discount_factor = discount_factor
    self.max_gradient    = max_gradient
    self.reg_param       = reg_param
    self.load_model      = load_model
    self. if_train       =  if_train   


    self.actor_network   = self.actor_network_struct
    self.critic_network  = self.critic_network_struct

    # exploration parameters
    self.exploration  = init_exp
    self.init_exp     = init_exp
    self.final_exp    = final_exp
    self.anneal_steps = anneal_steps

    # counters
    self.train_iteration = 0

    # rollout buffer
    self.state_buffer  = []
    self.reward_buffer = []
    self.action_buffer = []


    # model Settings
    self.device_count = 6
    self.policy_filter_num = 20
    self.input_filter_num = 7
    self.rl_inp_shape = (None,16, 16, self.input_filter_num*self.device_count)
    self.kernel_size = (3, 3)
    self.pool_size = (2, 2)

    #Saver


    # create and initialize variables
    self.create_variables()
    var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    uninitialized_vars = []
    # for var in tf.global_variables():
    for var in var_lists:
        try:
            # print(var)
            self.session.run(var)
        except tf.errors.FailedPreconditionError:
            # print("no - ", var)
            uninitialized_vars.append(var)


    self.init_new_vars_op = uninitialized_vars

    # var_lists = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    self.session.run(tf.variables_initializer(self.init_new_vars_op))

    # make sure all variables are initialized
    # self.session.run(tf.assert_variables_initialized())

    if self.summary_writer is not None:
      # graph was not available when journalist was created
      self.summary_writer.add_graph(self.session.graph)
      self.summary_every = summary_every

    self.saver = tf.train.Saver(var_list=self.init_new_vars_op)

    if self.load_model==1:
        self.saver.restore(self.session, "./policy_weights.ckpt")
        print("Model restored.")


  def actor_network_struct(self, states):
    # Convolutional Layer #1 and Pooling Layer #1
    conv1 = tf.layers.conv2d(
        inputs=states,
        filters=self.policy_filter_num,
        kernel_size=self.kernel_size ,
        padding="same",
        activation=tf.nn.relu)        
    
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=self.pool_size,strides=1)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=self.policy_filter_num,
        kernel_size=self.kernel_size ,
        padding="same",
        activation=tf.nn.relu)        
    
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=self.pool_size,strides=1)
    p2_shape=pool2.shape

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, p2_shape[1] * p2_shape[2] * p2_shape[3]])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    # dense = tf.layers.dense(inputs=dense, units=100, activation=tf.nn.relu)
    # dropout = tf.layers.dropout(
    #     inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    dropout = dense

    actor_out = tf.layers.dense(inputs=dropout, units=self.num_actions,activation=tf.nn.sigmoid)
    # self.actor_out = tf.layers.dense(inputs=dropout, units=self.num_actions,activation=tf.nn.softmax)
    return actor_out

  def critic_network_struct(self,states):

    conv1 = tf.layers.conv2d(
        inputs=states,
        filters=self.policy_filter_num,
        kernel_size=self.kernel_size ,
        padding="same",
        activation=tf.nn.relu)        
    
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=self.pool_size,strides=1)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=self.policy_filter_num,
        kernel_size=self.kernel_size ,
        padding="same",
        activation=tf.nn.relu)        
    
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=self.pool_size,strides=1)
    p2_shape=pool2.shape

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, p2_shape[1] * p2_shape[2] * p2_shape[3]])
    dense = tf.layers.dense(inputs=pool2_flat, units=100, activation=tf.nn.relu)
    # dense = tf.layers.dense(inputs=dense, units=100, activation=tf.nn.relu)
    # dropout = tf.layers.dropout(
    #     inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    dropout = dense

    critic_out = tf.layers.dense(inputs=dropout, units=1,activation=tf.nn.sigmoid)
    # self.actor_out = tf.layers.dense(inputs=dropout, units=self.num_actions,activation=tf.nn.softmax)
    return critic_out

  def resetModel(self):
    self.cleanUp()
    self.train_iteration = 0
    self.exploration     = self.init_exp
    # var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    # var_lists = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # self.session.run(tf.variables_initializer(var_lists))
    self.session.run(tf.variables_initializer(self.init_new_vars_op))

  def create_variables(self):

    with tf.name_scope("model_inputs"):
      # raw state representation
    #   print(self.state_dim)
    #   exit()
    #   self.states = tf.placeholder(tf.float32, shape=[None, self.state_dim], name="states")
      self.states = tf.placeholder(tf.float32, shape=(None, 16,16, 42), name="states")

    # rollout action based on current policy
    with tf.name_scope("predict_actions"):
      # initialize actor-critic network
      with tf.variable_scope("actor_network"):
        self.policy_outputs = self.actor_network(self.states)
      with tf.variable_scope("critic_network"):
        self.value_outputs = self.critic_network(self.states)

      # predict actions from policy network
      self.action_scores = tf.identity(self.policy_outputs, name="action_scores")

    # get variable list
    actor_network_variables  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor_network")
    critic_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic_network")

    # compute loss and gradients
    with tf.name_scope("compute_pg_gradients"):
      # gradients for selecting action from policy network
      self.taken_actions = tf.placeholder(tf.int32, shape=[None,self.num_actions], name="taken_actions")
      self.discounted_rewards = tf.placeholder(tf.float32, shape=[None,1], name="discounted_rewards")

      with tf.variable_scope("actor_network", reuse=True):
        self.logprobs = tf.log(self.actor_network(self.states))

      with tf.variable_scope("critic_network", reuse=True):
        self.estimated_values = self.critic_network(self.states)
      

      # compute policy loss and regularization loss
      self.cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logprobs, labels=self.taken_actions)
      self.pg_loss            = tf.reduce_mean(self.cross_entropy_loss)
      self.actor_reg_loss     = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in actor_network_variables])
      self.actor_loss         = self.pg_loss + self.reg_param * self.actor_reg_loss

      # compute actor gradients
      self.actor_gradients = self.optimizer.compute_gradients(self.actor_loss, actor_network_variables)
      # compute advantages A(s) = R - V(s)
    #   self.advantages = tf.reduce_sum(self.discounted_rewards - self.estimated_values)
      self.advantages = tf.reduce_sum(self.discounted_rewards - self.estimated_values)
      # compute policy gradients
      for i, (grad, var) in enumerate(self.actor_gradients):
        if grad is not None:
          self.actor_gradients[i] = (grad * self.advantages, var)

      # compute critic gradients
      self.mean_square_loss = tf.reduce_mean(tf.square(self.discounted_rewards - self.estimated_values))
      self.critic_reg_loss  = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in critic_network_variables])
      self.critic_loss      = self.mean_square_loss + self.reg_param * self.critic_reg_loss
      self.critic_gradients = self.optimizer.compute_gradients(self.critic_loss, critic_network_variables)

      # collect all gradients
      self.gradients = self.actor_gradients + self.critic_gradients

      # clip gradients
      for i, (grad, var) in enumerate(self.gradients):
        # clip gradients by norm
        if grad is not None:
          self.gradients[i] = (tf.clip_by_norm(grad, self.max_gradient), var)

      # summarize gradients
      for grad, var in self.gradients:
        tf.summary.histogram(var.name, var)
        if grad is not None:
          tf.summary.histogram(var.name + '/gradients', grad)

      # emit summaries
      tf.summary.histogram("estimated_values", self.estimated_values)
      tf.summary.scalar("actor_loss", self.actor_loss)
      tf.summary.scalar("critic_loss", self.critic_loss)
      tf.summary.scalar("reg_loss", self.actor_reg_loss + self.critic_reg_loss)

    # training update
    with tf.name_scope("train_actor_critic"):
      # apply gradients to update actor network
      self.train_op = self.optimizer.apply_gradients(self.gradients)

    self.summarize = tf.summary.merge_all()
    self.no_op = tf.no_op()

  def sampleAction(self, states_):

    actions_prob_v,state_value  = self.session.run([self.action_scores, self.estimated_values], {self.states: states_})

    explore_rand=np.random.uniform(low=0,high=1)

    if (explore_rand < self.exploration) and (self.if_train==1):
      action_probs=0.5*np.ones(shape=(1,self.device_count),dtype="float32")
    else:
      action_probs=actions_prob_v
  
    self.predicted_actions=np.random.binomial(n=1,p=action_probs)

    return self.predicted_actions[0], state_value[0],action_probs
  
  def updateModel(self):

    N = len(self.reward_buffer)
    r = 0 # use discounted reward to approximate Q value

    # compute discounted future rewards
    discounted_rewards = np.zeros(N)
    for t in reversed(range(N)):
      # future discounted reward from now on
      r = self.reward_buffer[t] + self.discount_factor * r
      discounted_rewards[t] = r
    #   print(discounted_rewards,'---,')

    # whether to calculate summaries
    calculate_summaries = self.summary_writer is not None and self.train_iteration % self.summary_every == 0

    # update policy network with the rollout in batches
    for t in range(N):

      # prepare inputs
      states_  = self.state_buffer[t][np.newaxis, :]
      actions_ = np.array([self.action_buffer[t]])
      rewards_ = np.array([discounted_rewards[t]])
      rewards_ = rewards_[np.newaxis,:]

      # perform one update of training
      _, cross_entropy_loss_v, mean_square_loss_v, summary_str = self.session.run([
        self.train_op, self.cross_entropy_loss, self.mean_square_loss,
        self.summarize if calculate_summaries else self.no_op], 
        {self.states:             states_,
        self.taken_actions:      actions_,
        self.discounted_rewards: rewards_})
    #   print("---3----")

      # emit summaries
      if calculate_summaries:
        self.summary_writer.add_summary(summary_str, self.train_iteration)

    self.annealExploration()
    self.train_iteration += 1

    # clean up
    self.cleanUp()

    return cross_entropy_loss_v, mean_square_loss_v

  def annealExploration(self, stategy='linear'):
    ratio = max((self.anneal_steps - self.train_iteration)/float(self.anneal_steps), 0)
    self.exploration = (self.init_exp - self.final_exp) * ratio + self.final_exp

  def storeRollout(self, state, action, reward):
    self.action_buffer.append(action)
    self.reward_buffer.append(reward)
    self.state_buffer.append(state)

  def cleanUp(self):
    self.state_buffer  = []
    self.reward_buffer = []
    self.action_buffer = []