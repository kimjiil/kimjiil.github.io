export const SITE_TITLE = "kimjiil.github.io";
export const SITE_DESC = "배운 것을 기록하는 개발 블로그";
export const AUTHOR = "Jiil Kim";
export const GITHUB_URL = "https://github.com/kimjiil";
export const EMAIL = "kimjiil2013@naver.com";

export interface NavItem {
  title: string;
  href: string;
  sublinks?: { title: string; href: string }[];
}

export const NAV_ITEMS: NavItem[] = [
  {
    title: "Paper Review",
    href: "/papers",
    sublinks: [
      { title: "Object Detection", href: "/papers/object-detection-paper" },
      { title: "Reinforcement Learning", href: "/papers/reinforcement-learning-paper" },
      { title: "Image Generation", href: "/papers/image-generation-paper" },
      { title: "Computer Vision", href: "/papers/computer-vision-paper" },
      { title: "Deep Learning", href: "/papers/deep-learning-paper" },
    ],
  },
  {
    title: "Study",
    href: "/study",
    sublinks: [
      { title: "Math", href: "/study/math" },
      { title: "AI/ML", href: "/study/ai-ml" },
      { title: "Algorithm", href: "/study/algorithm" },
      { title: "Game", href: "/study/game" },
      { title: "etc", href: "/study/etc" },
    ],
  },
  {
    title: "Projects",
    href: "/projects",
    sublinks: [
      { title: "Object Detection", href: "/projects/object-detection-project" },
      { title: "Reinforcement Learning", href: "/projects/reinforcement-learning-project" },
    ],
  },
  { title: "Tags", href: "/tags" },
  { title: "About", href: "/about" },
];

// 카테고리 정의 (옛 Jekyll 사이트 메뉴 구조)
export const CATEGORIES: Record<string, { section: string; label: string }> = {
  "object-detection-paper": { section: "papers", label: "Object Detection" },
  "reinforcement-learning-paper": { section: "papers", label: "Reinforcement Learning" },
  "image-generation-paper": { section: "papers", label: "Image Generation" },
  "computer-vision-paper": { section: "papers", label: "Computer Vision" },
  "deep-learning-paper": { section: "papers", label: "Deep Learning" },
  "math": { section: "study", label: "Math" },
  "ai-ml": { section: "study", label: "AI/ML" },
  "algorithm": { section: "study", label: "Algorithm" },
  "game": { section: "study", label: "Game" },
  "etc": { section: "study", label: "etc" },
  "object-detection-project": { section: "projects", label: "Object Detection" },
  "reinforcement-learning-project": { section: "projects", label: "Reinforcement Learning" },
};

export const SECTIONS: Record<string, { title: string; desc: string }> = {
  papers: { title: "Paper Review", desc: "AI/ML 논문을 읽고 정리한 리뷰" },
  study: { title: "Study", desc: "공부하며 기록한 노트" },
  projects: { title: "Projects", desc: "직접 만들어본 것들" },
};
