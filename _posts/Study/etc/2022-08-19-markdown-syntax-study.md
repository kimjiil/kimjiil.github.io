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

## 제목

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