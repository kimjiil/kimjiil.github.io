// 그래프 데이터 구성 헬퍼 — /graph(전체)와 포스트 로컬 그래프에서 공용
import type { CollectionEntry } from "astro:content";
import { CATEGORIES, SECTIONS } from "../consts";

export type GNode = {
  id: string;
  label: string;
  kind: "section" | "category" | "post";
  section: string;
  url?: string;
};
export type GEdge = { source: string; target: string; kind: "tree" | "tag" };
export type GraphData = { nodes: GNode[]; edges: GEdge[] };

const slugOf = (p: CollectionEntry<"posts">) => p.id.split("/").pop()!;

/** 전체 그래프: 섹션 → 카테고리(포스트 있는 것만) → 포스트 + 태그 공유 엣지 */
export function buildGraphData(posts: CollectionEntry<"posts">[]): GraphData {
  const nodes: GNode[] = [];
  const edges: GEdge[] = [];

  for (const [key, s] of Object.entries(SECTIONS)) {
    nodes.push({ id: `s:${key}`, label: s.title, kind: "section", section: key, url: `/${key}` });
  }

  const usedCats = new Set(posts.map((p) => p.data.category));
  for (const [key, c] of Object.entries(CATEGORIES)) {
    if (!usedCats.has(key)) continue;
    nodes.push({ id: `c:${key}`, label: c.label, kind: "category", section: c.section, url: `/${c.section}/${key}` });
    edges.push({ source: `s:${c.section}`, target: `c:${key}`, kind: "tree" });
  }

  for (const p of posts) {
    const cat = CATEGORIES[p.data.category];
    const slug = slugOf(p);
    nodes.push({
      id: `p:${slug}`,
      label: p.data.title,
      kind: "post",
      section: cat?.section ?? "study",
      url: `/posts/${slug}/`,
    });
    edges.push({ source: `c:${p.data.category}`, target: `p:${slug}`, kind: "tree" });
  }

  for (let i = 0; i < posts.length; i++) {
    for (let j = i + 1; j < posts.length; j++) {
      const a = posts[i]!;
      const b = posts[j]!;
      if (a.data.tags.some((t) => b.data.tags.includes(t))) {
        edges.push({ source: `p:${slugOf(a)}`, target: `p:${slugOf(b)}`, kind: "tag" });
      }
    }
  }

  return { nodes, edges };
}

/** 로컬 그래프: 현재 포스트 + 소속 카테고리/섹션 + 태그 공유 이웃 (depth 1) */
export function buildLocalGraphData(posts: CollectionEntry<"posts">[], currentSlug: string): GraphData {
  const full = buildGraphData(posts);
  const centerId = `p:${currentSlug}`;

  const keep = new Set<string>([centerId]);
  for (const e of full.edges) {
    if (e.source === centerId) keep.add(e.target);
    if (e.target === centerId) keep.add(e.source);
  }
  // 소속 카테고리의 상위 섹션도 포함
  for (const e of full.edges) {
    if (e.kind === "tree" && keep.has(e.target) && e.source.startsWith("s:")) keep.add(e.source);
  }

  return {
    nodes: full.nodes.filter((n) => keep.has(n.id)),
    edges: full.edges.filter((e) => keep.has(e.source) && keep.has(e.target)),
  };
}
