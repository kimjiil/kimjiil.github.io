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

디자인은 [sudoremove.com](https://sudoremove.com) ([sudormrf-run/web](https://github.com/sudormrf-run/web), © 2026 Jong Hyun Park, MIT-NC License)의
터미널 다크 테마 스타일을 참고/변형했습니다. 이 블로그는 비상업적 개인 블로그입니다.
