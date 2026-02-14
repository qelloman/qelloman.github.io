import { parseFrontmatter } from '../utils/frontmatter';

export interface FSNode {
  name: string;
  type: 'file' | 'directory';
  content?: string;
  date?: string;
  children?: FSNode[];
}

function sortByDateDesc(nodes: FSNode[]): FSNode[] {
  return [...nodes].sort((a, b) => {
    const da = a.date ?? '0000-00-00';
    const db = b.date ?? '0000-00-00';
    return db.localeCompare(da);
  });
}

// Auto-import all .md files from content/
const allMdFiles = import.meta.glob('/content/**/*.md', { eager: true, query: '?raw', import: 'default' }) as Record<string, string>;

interface DirMap {
  [key: string]: FSNode[];
}

function buildFilesystem(): FSNode {
  const dirs: DirMap = {};
  const rootFiles: FSNode[] = [];

  for (const [path, raw] of Object.entries(allMdFiles)) {
    // path looks like "/content/experience/samsung.md"
    const relative = path.replace('/content/', '');
    const parts = relative.split('/');
    const filename = parts[parts.length - 1];
    const { meta, content } = parseFrontmatter(raw);

    const node: FSNode = {
      name: filename,
      type: 'file',
      content,
      date: meta.date,
    };

    if (parts.length === 1) {
      // root-level file
      rootFiles.push(node);
    } else {
      // file inside a subdirectory
      const dirName = parts[0];
      if (!dirs[dirName]) dirs[dirName] = [];
      dirs[dirName].push(node);
    }
  }

  const dirNodes: FSNode[] = Object.entries(dirs).map(([name, files]) => ({
    name,
    type: 'directory' as const,
    children: sortByDateDesc(files),
  }));

  // Sort directories alphabetically
  dirNodes.sort((a, b) => a.name.localeCompare(b.name));

  return {
    name: '~',
    type: 'directory',
    children: [...dirNodes, ...sortByDateDesc(rootFiles)],
  };
}

export const filesystem = buildFilesystem();

export function resolvePath(path: string[], root: FSNode = filesystem): FSNode | null {
  let current = root;
  for (const segment of path) {
    if (segment === '~' || segment === '') continue;
    if (current.type !== 'directory' || !current.children) return null;
    const child = current.children.find((c) => c.name === segment);
    if (!child) return null;
    current = child;
  }
  return current;
}

export function getVisibleEntries(node: FSNode): FSNode[] {
  return node.children ?? [];
}

export function getPathString(path: string[]): string {
  if (path.length === 0) return '~';
  return '~/' + path.join('/');
}
