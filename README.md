# kimjiil.github.io

개인 블로그. [Astro](https://astro.build) 기반, 터미널 다크 테마.

## 개발

```bash
npm install
npm run dev      # http://localhost:4321
npm run build    # dist/ 에 정적 빌드
```

## 구조

- `src/content/posts/{papers,study,projects}/` — 마크다운 포스트 (frontmatter: title, date, category, tags)
- `src/consts.ts` — 네비게이션/카테고리 정의
- `public/images/` — 포스트 이미지
- master 브랜치 push 시 GitHub Actions로 자동 배포

## Credits

초기 구축 시 [sudoremove.com](https://sudoremove.com) ([sudormrf-run/web](https://github.com/sudormrf-run/web), © 2026 Jong Hyun Park)의
터미널 다크 테마에서 영감을 받았습니다. 이후 컬러 팔레트(네이비+시안)와 레이아웃, 인터랙티브 데스크 씬,
그래프 뷰 등 디자인·코드 전반을 독자적으로 재구성했으며, 해당 저장소의 코드는 사용하지 않았습니다.
