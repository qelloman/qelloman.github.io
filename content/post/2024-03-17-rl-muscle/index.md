---
title: "(WIP) Building my RL muscle"
date: 2024-03-17T21:45:16+09:00
math: true
---

One of my goals in this year is to submit AI conference paper. (NOTE that it is not about getting accepted.) Personally, I am pretty interested in reinforcement learning (RL) even though it takes so much math background, which I do not have for now. While I am reading many RL papers to find research topics to write, I am writing RL codes at the same time.

Of course there are tons of code out there I can simply copy and paste to do experiment about well-known environments such as MuJoCo. BUT, one of my friends told me that there is a concept called coding muscle. You can basically build your own coding muscle by writing numerous lines of code. Also, it is totally different to understand RL papers with and without writing code. Therefore, I deciede to write my own RL code from the simpliest one.


# SARSA vs Q-Learning (3/10)

If you just started an RL journey, then you got to learn about fundamental concepts like MDP or Bellman equations. At some point, you will get to meet the first RL-like algorithm which are SARSA and Q-learning.

Both use the Bellman equation to update Q-values. Also they use epsilon greedy for training. However, one of the biggest different between them is that Q-learning uses the maximum Q-value of the next state for target Q-value. This brings two major differences.
1. Q-learning becomes an off-policy algorithm. Although the agent did not take an action which give the maximum Q-value (behavioral), the update uses the maximum Q-value (target).
2. Q-learning becomes more adventurous and optimistic. For exmaple, a state could have multiple actions. The average Q-value over the state's actions is low. However, if one action has very large Q-value, then the agent would try to get into the stage. This behavior is demonstrated well by Cliff walking environment.

## Cliff walking
There is a grid environment. You need to walk one end to the other end as with few step as possible. However, if you step onto the cliff, then you will die. It is the trade-off between risk and the number of steps. If you take risk, then you can take the shortest path. On the other hand, if you really don't want to die, then you always detour. Q-learning acts like the former one whereas SARSA acts like the latter one.

## Implementation
My impression on the implementation of SARSA, Q-Learning, and Cliff Walking is easy. Basically, the environment for cliff walking is provided by gymnasium. SARSA and Q-Learning are not difficult algorithms if you undersatnd Bellman equation. Since the cliff walking env does not have many states, we can simply use table to store Q-value. I wrote and ran the code. Epic fail.

I did make a huge mistake. I put `state` in the conditional statement to check if we can update Q-value or not. However, one of our `state`s is `0` so the state's Q-value is never updated. For the worse, the `0` state is critical state for SARSA algorithm because it is on the corner. So SARSA never went through the corner tile and stuck there.

I found out this stupid mistake after 3 hours of debugging. What a shame. Anyway, I'd liked to cheer for myself because I really felt like I built tiny muscles. It was very sore.


