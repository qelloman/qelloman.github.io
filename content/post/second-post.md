---
title: "Hugo + github page 셋업 하기"
date: 2023-04-29T16:14:55+09:00
---

개인 개발 블로그를 세팅 해야지 해야지 하면서 너무 뒤로만 미뤄뒀다가 굉장히 멋져보이는 블로그를 발견해서 벤치마크하기로 했다.
Hugo라는 개인 블로그 프레임워크(?)와 github page 조합으로 블로그를 시작하려할 때 최소한의 스텝으로 어떻게 시작하는지 알아보자.

(참고로 Mac 환경을 기준으로 설명하는 것이다.)

# 설치 방법

1. Hugo 설치하기

``` shell
brew install hugo
```


2. Hugo를 통해 웹사이트 만들기

``` shell
hugo new site qelloman-github-io
```
위와 같은 명령어를 이용하면 qelloman-github-io라는 디렉토리가 생길 것이고, 여기서 웹사이트의 세팅이나 새로운 포스트를 만들 수 있다.
(`qelloman-github-io` 디렉토리에 웹페이지 파일은 없을 것이다. 이는 나중에 생긴다.)

3. git init 해주기

```
cd qelloman-github-io
git init
```
다른 설명들을 보면 hugo project 디렉토리(여기서는 `qelloman-github-io`)와 실제 웹사이트가 담겨져 있는 디렉토리(보통 `public`이라는 하위 디렉토리)를 둘 다 github에 올리는 경우가 있는데 최신 버전은 hugo project 디렉토리만 올린다.

4. 테마 받아오기

나는 papermod라는 테마가 마음에 들어서 사용하였다. `theme` 아래에 아래와 같은 명령어를 사용하여 마음에 드는 테마를 가져오자. 직접 클론하는 방법도 있겠지만 hugo에서는 submodule을 사용하기를 권장한다.
```
git submodule add --depth=1 https://github.com/adityatelange/hugo-PaperMod.git themes/PaperMod
git submodule update --init --recursive # needed when you reclone your repo (submodules may not get cloned automatically)
```

5. config.yml 파일 수정

예전에는 config.toml이라는 파일을 사용했던 것 같은데 현재는 config.yaml 파일로 통일되어 있다. `PaperMod`[LINK](https://github.com/adityatelange/hugo-PaperMod) 테마 깃헙에는 config.yaml 파일의 예시가 나와있어서 복사해서 붙였다.

일단 title을 바꿔주고, theme를 지정해야 한다.
``` yaml
baseURL: "http://qelloman.github.io/"
title: Qel'Log
paginate: 5
theme: PaperMod
```

6. github에 repository 만들기

나 같은 경우는 qelloman-github-io라는 github repository를 만들었다. 반드시 public으로 만들어야 한다.

7. github action 만들기

빌드할 때마다 자동으로 배포하기 위한 workflow를 지원하기 위해 workflow 파일을 만들어줘야 한다.

`.github/workflows/hugo.yaml`이라는 파일을 만들고 다음과 같은 내용을 입력해 준다. (Hugo 공식 홈페이지 참조 [LINK](https://gohugo.io/hosting-and-deployment/hosting-on-github/))
``` yaml
# Sample workflow for building and deploying a Hugo site to GitHub Pages
name: Deploy Hugo site to Pages

on:
  # Runs on pushes targeting the default branch
  push:
    branches:
      - main

  # Allows you to run this workflow manually from the Actions tab
  workflow_dispatch:

# Sets permissions of the GITHUB_TOKEN to allow deployment to GitHub Pages
permissions:
  contents: read
  pages: write
  id-token: write

# Allow only one concurrent deployment, skipping runs queued between the run in-progress and latest queued.
# However, do NOT cancel in-progress runs as we want to allow these production deployments to complete.
concurrency:
  group: "pages"
  cancel-in-progress: false

# Default to bash
defaults:
  run:
    shell: bash

jobs:
  # Build job
  build:
    runs-on: ubuntu-latest
    env:
      HUGO_VERSION: 0.111.3
    steps:
      - name: Install Hugo CLI
        run: |
          wget -O ${{ runner.temp }}/hugo.deb https://github.com/gohugoio/hugo/releases/download/v${HUGO_VERSION}/hugo_extended_${HUGO_VERSION}_linux-amd64.deb \
          && sudo dpkg -i ${{ runner.temp }}/hugo.deb          
      - name: Install Dart Sass Embedded
        run: sudo snap install dart-sass-embedded
      - name: Checkout
        uses: actions/checkout@v3
        with:
          submodules: recursive
          fetch-depth: 0
      - name: Setup Pages
        id: pages
        uses: actions/configure-pages@v3
      - name: Install Node.js dependencies
        run: "[[ -f package-lock.json || -f npm-shrinkwrap.json ]] && npm ci || true"
      - name: Build with Hugo
        env:
          # For maximum backward compatibility with Hugo modules
          HUGO_ENVIRONMENT: production
          HUGO_ENV: production
        run: |
          hugo \
            --gc \
            --minify \
            --baseURL "${{ steps.pages.outputs.base_url }}/"          
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v1
        with:
          path: ./public

  # Deployment job
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v2

```

8. test용 포스트 만들기

다음과 같은 명령어를 이용해 포스트를 만들 수 있다.
```
hugo new --kind post test.md
```

이렇게 하고 나면 `content/post/test.md`라는 파일이 생성이 되었을 것이다. 가서 제목을 바꿔주어도 좋고, 내용을 추가해도 좋다. 그리고 draft mode가 기본적으로 켜져 있기 때문에 test.md 파일에서 삭제해 주어야 한다.

9. 동작 확인

다음과 같은 명령어로 로컬에서 웹사이트를 호스팅할 수 있다.
```
hugo server
```

그러면 http://localhost:1313/에서 웹사이트를 확인할 수 있다. (draft 모드인 글까지 보고 싶으면 뒤에 -D 옵션을 붙이면 된다.)

10. 로컬 repository를 remote로 푸쉬

로컬에 있는 디렉토리를 push한다. 그래도 푸쉬하면 workflow 항목이 있기 때문에 토큰에 관련된 에러 메세지가 나오면서 되지 않는다. 이를 해결하기 위해 github에서 settings - developer settings - personal access tokens에 들어가서 토큰이 없다면 만들고 workflow를 체크해 주어야 한다.

```
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/qelloman/qelloman-github-io.git
git push -u origin main
```

11. page 세팅 및 action 확인

push가 성공적으로 되었다면 github repo 세팅을 바꿔서 웹사이트를 호스팅해야 한다.

repo의 [setting] - [pages]로 들어가서 [Build and deployment]에서 [Source]를 GitHub Actions로 바꿔준다.

그리고 나서 Actions 항목으로 가면 `page build and development`라는 웹사이트를 빌드하기 위한 action과 우리가 추가해준 workflow가 열심히 돌아가는 것을 확인할 수 있을 것이다.


# 참고자료

https://www.youtube.com/watch?v=psyz4UPnGAA&ab_channel=CodeNanshu
