---
title: "Markdown 문법 모음"
categories:
  - etc
tags:
  - markdown
  - html
date: 2022-08-19-13:40:00
toc: true
toc_sticky: true
toc_icon: "sticky-note"
use_math: true
---

## Github io 변경 사항
### markdown code block test

```terminal
TEST~~TEST~~TEST~~TEST~~TEST~~TEST~~TEST~~TEST~~
TEST~~
TEST~~TEST~~TEST~~
TEST~~TEST~~TEST~~TEST~~TEST~~TEST~~TEST~~TEST~~
```

```python
def fun(**kwargs):
    pass

class test:
    def __init__(self):
        test = 0

```

```cpp
template <typename T>
void sort(T start, T end, Compare comp);
#include<iostream>
#include<algorithm>
using namespace std;
void Print(int *arr)
{
    cout << "arr[i] : ";
}

```

```java
public static int test(String[] a)
{
    int a = 0;
    System.out.println("test");
  
    return a;
}

```


```markdown 
Test~~
function a() {
    test();
}
```

```javascript
fucntion test() {
    console.log("gg", no);
}
```

### Code Block 라인 줄 표시
수정 파일 - _config.yml
```yaml
markdown: kramdown
kramdown:
...
  syntax_highlighter: rouge
  syntax_highlighter_opts:
    block:
        line_numbers: true
```

라인 줄 복사시 라인 번호도 같이 복사되는 현상 방지
수정 파일 - _sass/minimal-mistakes/_syntax.scss
```scss
/* line numbers*/
    &.gutter,
    &.rouge-gutter {
      padding-right: 1em;
      width: 1em;
      color: $base04;
      border-right: 1px solid $base04;
      text-align: right;
      // 라인이 복사되지 않게 한다.
      -webkit-touch-callout: none;
      -webkit-user-select: none;
      -khtml-user-select: none;
      -moz-user-select: none;
      -ms-user-select: none;
      user-select: none;
    }

```

### Code Block Sytle 변경
수정 파일 - _sass/minimal-mistakes.scss
하단에 다음 코드 추가
```scss
...
@import "minimal-mistakes/code_style"

```

_sass/minimal-mistakes/ 폴더에 _code_style.scss 파일 추가

각 문법에 해당하는 코드를 추가 (예시 yaml)
```scss
$window-height: 16px;

.language-yaml {
  position: relative;
  margin-bottom: 1.5em;
  padding: calc(#{$window-height * 0.3} + 0em) 0em 0em;
  border: 1px solid $border-color;
  border-radius: $border-radius;
  box-shadow: 0 0.25em 1em rgba($text-color, 0.25); //base color
  background-color: $background-color; //base color

  &::before {
    content: "yaml";
    position: absolute;
    top: 0;
    left: 0;
    margin: 0;
    padding: 0 0;
    background: mix($background-color, #fff, 25%); //base color
    color: mix($text-color, #FFBF00, 50%); //base color
    font-size: ($window-height);
    line-height: 0;
    text-indent: (0.5 * $window-height);
  }

  .highlight {
    margin: 0;
    padding: 0;
    background-color: initial;
    color: #fff;
  }
}
```

### Post 하단에 Date 정보 출력

수정 파일 - _config.yml
```yaml
...
# Defaults
defaults:
  # _posts
  - scope:
      path: ""
      type: posts
    values:
      layout: single
      author_profile: true
      read_time: false # 이 부분을 false 수정
      show_date: true # 추가 부분
      share: true
      related: true
```