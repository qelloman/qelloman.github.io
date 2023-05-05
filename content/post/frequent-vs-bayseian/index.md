---
title: "Bayesian Statistics"
date: 2023-05-01T19:43:25+09:00
---

Bayesian 개념이 나오면 Bayes' Rule에 대해서만 알고 그 근본적인 개념에 대해 계속 헷갈리게 되서 이번 기회에 한번 정리해 보기로 한다.

# 베이스주의 (Bayesian) 그리고 빈도주의 (Frequentist)

Wikipedia에서 정리한 Bayesian statistics의 정의에 대해서 알아보자.

> Bayesian statistics is a theory in the field of statistics based on the Bayesian interpretation of probability where probability expresses a degree of belief in an event. The degree of belief may be based on prior knowledge about the event, such as the results of previous experiments, or on personal beliefs about the event. This differs from a number of other interpretations of probability, such as the frequentist interpretation that views probability as the limit of the relative frequency of an event after many trials.[1]

위에서 말한 것처럼 Bayesian statistics에서는 확률을 보는 관점이 "어떤 사건의 **Degree of belief**"이다. 즉 어떤 사건이 일어날 것이라고 얼마나 확신하냐를 확률로 본다. (생각해보면 고등학교 통계 때는 확률이 어떤 의미를 지니는지 깊게 생각하지 않았던 것 같다.)

이와 반대로 빈도주의는 좀 더 단순하게 어떤 사건이 무한대로 시행이 됐을 때 몇번이나 사건이 발생할지를 말한다.

Bayesian statistics는 굉장히 오래전부터 있었지만 철학적, 그리고 실질적인 이유로 많이 안쓰였고 20세기는 대부분 빈도주의로 확률을 접근했다고 한다. (뒤에서도 이야기 하겠지만 새로운 데이터가 주어질 때마다 어떤 값들이 변하는 것은 계산이 많고 복잡하니 그랬던 것 같다.) 하지만 이제 계산 능력이 발전하고 Markov chain Monte Carlo 같은 새로운 알고리즘을 사용하면서 21세기에는 많이 쓰인다고 한다.

# Bayes' theorem

A와 B라는 사건이 있을 때, B가 일어났을 때, A의 조건부 확률은 다음과 같이 주어진다.

$$ P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

- $A$는 보통 proposition (명제)를 나타낸다. 예를 들면 동전을 던지면 반은 앞면이 나온다.
- $B$는 evidence를 나타낸다. 우리가 고려해야 되는 데이터를 말한다. 
- $P(A)$는 `prior probability`인데 $B$가 주어지기 이전에 우리가 기존에 믿고 있던 belief를 나타낸다.
- $P(B|A)$는 `likelihood function`으로 A가 참일 때 B라는 데이터가 나올 확률을 의미한다. 즉, 데이터 B가 명제 A를 얼마나 support하는지를 수치적으로 나타낸다.
- 마지막으로 $P(A|B)$는 `posterior probability`로 B를 고려했을 때 명제 A에 대한 belief를 나타낸다.

> 참고: $P(B)$는 scaling factor로써 integral을 해야해서 구하기 어렵기도 하고, maximum a posterior (MAP)를 하기 위한 optimization technique에서는 계산할 필요가 없다.

Bayesian은 결국 `prior`에서 새로운 데이터 $B$가 주어졌을 때 `posterior`로 업데이트해가는 과정이다. 이와 반대로 기존의 빈도주의의 statistics에서는 업데이트가 아니라 주어진 데이터로 **우리가 불변이라고 믿고, 참이라고 믿고 있는** 모수의 확률분포에 대한 파라미터들을 한번 구하고 끝난다.

# Bayesian Inference
위에서 설명한대로 classical한 statistics(frequentist inference)에서는 모델 파라미터와 가정이 고정되어있다고 생각한다. 이와 달리 Bayesian inference는 모델 파라미터가 바뀔 수 있다고 생각한다. (보통 모델 파라미터 또한 확률 변수라고 생각한다.) 그리고 데이터에 따라 모델 파라미터를 업데이트 해준다.

동전 던지기의 예를 들어보자.

동전을 던졌는데 4번이 앞면이 나왔다면, Frequentist와 Bayesian은 어떻게 생각할까?

## Frequentist

Frequentist는 그냥 앞면이 나올 빈도수가 4/4였으므로 100% 앞면이 나오는 동전이라고 결론을 지을 것이다. Frequentist는 데이터를 가지고 절대적이고 불변인 확률 분포의 파라미터를 결정하게 되는데 지금 우리가 얻은 데이터를 가지고는 앞면 밖에 못봤으니 그냥 앞면이 나올 확률 $\theta=1$이라고 생각할 것이다.

### Maximum Likellihood estimation

여기서 주어진 데이터가 나올 확률이 가장 큰 $\theta$를 MLE를 통해서 구해보면 
$$L(\theta, x=H,H,H,H)=\theta^4$$
$$\theta_{MLE}=\argmax_\theta \theta^4 = 1$$

## Bayesian

반면, Bayesian은 기존의 belif를 활용한다. (`prior`) 기존에 나는 동전이 앞면이 나올 확률(Bayesian 관점에서)이 $p=0.5$인 베르누이 분포를 따른다고 가정했다. 그러면 새롭게 4개가 전부 앞면이 나온 데이터를 보고 $p$를 업데이트하게 된다.

- Bayes Theorem: $$P(\theta | x)=\frac{P(x|\theta)p(\theta)}{P(x)}$$
- Likelihood (Bernoulli Likelihood): $$ P(x|\theta)=\theta^x (1-\theta)^{1-x}$$
- Prior (Beta): $$P(\theta | \alpha, \beta)=\frac{\theta^{\alpha -1} (1-\theta)^{\beta-1}}{B(\alpha, \beta)}$$
- Posterior (Beta): $$P(\theta|x)=\frac{\theta^x (1-\theta)^{1-x}\theta^{\alpha -1} (1-\theta)^{\beta-1}}{P(x)B(\alpha, \beta)} \propto \theta^{\alpha+x-1}(1-\theta)^{\beta+(1-x)-1}=Beta(\hat{\alpha}, \hat{\beta}$$

즉, 말로써 풀어보면 우리는 Beta distribution을 통해 $\theta$(여기서는 동전이 앞면이 나올 확률 파라미터)를 나타낸다. 이 때 x가 어떻게 나오는지에 따라서 $\theta$의 확률 분포의 파라미터인 $\alpha$와 $\beta$를 업데이트하게 된다.

> prior와 posterior가 같은 분포를 띄도록 만드는 prior를 `conjugate prior`라고 부른다.

$\alpha=2$, $\beta=2$라면 $\theta=0.5$에서 probability가 최대치를 갖는 prior를 갖게 된다. 이 때 위처럼 4개가 전부 앞면이 나온다면

$$P(\theta|x)\propto \theta^{2+4-1}(1-\theta)^{2+0-1}=Beta(6,2)$$

$\alpha=2$, $\beta=2$인 $Beta$ 분포에서  $\hat{\alpha}=6$, $\hat{\beta}=2$인 $Beta$ 분포로 바뀌었다.

> Bayesian inference에서 $\theta$가 고정된 값이 아니라 확률 분포 형태로 나왔다는 것에 주목하자.

### Maximum A Posterior (MAP)

$\theta$를 위에서 구한 prior를 가지고 MAP을 이용해서 구해보면 다음과 같다.
$$\theta_{MAP}=\argmax_\theta {\theta^5 (1-\theta)}=\frac{5}{6}$$


> 참고로 데이터가 무수히 많이 주어진다면 MAP과 MLE는 수렴하게 될 것이다.

# 결론

딥러닝에서도 많이 보게될 Bayesian statistics의 개념과 Bayesian inference가 어떻게 이뤄지는지 살펴봤다. Bayesian inference의 경우 모수의 확률분포를 나타내는 파라미터를 고정된 값이 아닌 확률분포로 나타내고, 이러한 확률 분포는 prior로 bayes theorem에 적용된다. 그리고 데이터를 통해서 posterior를 구하고 posterior를 최대화 시키는 파라미터로 정한다.

# 참고자료
[1] Bayesian statistics wiki - https://en.wikipedia.org/wiki/Bayesian_statistics

[2] Bayesian inference wiki - https://en.wikipedia.org/wiki/Bayesian_inference

[3] Maximum Likelihood StatQuest -  https://www.youtube.com/watch?v=XepXtl9YKwc&ab_channel=StatQuestwithJoshStarmer

[4] 베르누이 분포 최대 가능도 추정법 - https://datascienceschool.net/02%20mathematics/09.02%20%EC%B5%9C%EB%8C%80%EA%B0%80%EB%8A%A5%EB%8F%84%20%EC%B6%94%EC%A0%95%EB%B2%95.html

[5] 베이즈 추정법 - https://datascienceschool.net/02%20mathematics/09.03%20%EB%B2%A0%EC%9D%B4%EC%A6%88%20%EC%B6%94%EC%A0%95%EB%B2%95.html

[6] 기초 베이지안 이론 - https://gaussian37.github.io/ml-concept-basic_bayesian_theory/
