---
title: "Variational Auto Encoder(VAE)와 latent vector"
date: 2023-05-06T23:07:44+09:00
draft: true
---

variational auto encoder를 공부하면서 가장 헷갈리는 부분은 이거였다.

- $P(z, x)$와 $P(x|z)$의 차이는 무엇인가? 실질적인 차이는 $P(x|z)$는 scaling factor인 $P(z)$가 들어간다는 점이다. 근데 이게 실제로 우리가 이해할 수 있는 의미는 무엇일까?