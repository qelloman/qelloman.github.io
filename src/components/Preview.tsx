import { theme } from '../theme';
import type { FSNode } from '../data/filesystem';
import { getPathString } from '../data/filesystem';
import { renderMarkdown } from '../utils/markdown';

interface PreviewProps {
  node: FSNode | null;
  path: string[];
  focused: boolean;
  previewSelectedIndex: number;
  onPreviewSelect: (index: number) => void;
  onPreviewOpen: (index: number) => void;
}

export function Preview({ node, path, focused, previewSelectedIndex, onPreviewSelect, onPreviewOpen }: PreviewProps) {
  const title = node
    ? node.type === 'file'
      ? getPathString([...path, node.name])
      : getPathString(path)
    : '~';

  return (
    <div style={{
      flex: 1,
      display: 'flex',
      flexDirection: 'column',
      background: theme.bg,
      minWidth: 0,
    }}>
      <div style={{
        padding: '6px 12px',
        fontSize: 12,
        color: focused ? theme.accent4 : theme.textDim,
        borderBottom: `1px solid ${theme.border}`,
        fontWeight: 600,
        letterSpacing: 1,
        textTransform: 'uppercase',
      }}>
        Preview — {title}
      </div>

      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '12px 20px',
        lineHeight: 1.6,
        fontSize: 14,
        color: theme.text,
      }}>
        {node?.type === 'file' && node.content ? (
          renderMarkdown(node.content)
        ) : node?.type === 'directory' ? (
          <DirectoryPreview
            node={node}
            path={path}
            focused={focused}
            selectedIndex={previewSelectedIndex}
            onSelect={onPreviewSelect}
            onOpen={onPreviewOpen}
          />
        ) : (
          <WelcomeScreen />
        )}
      </div>
    </div>
  );
}

function WelcomeScreen() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: 12 }}>
      <pre style={{ color: theme.accent3, fontSize: 12, lineHeight: 1.3 }}>{`
  ╔══════════════════════════════════╗
  ║                                  ║
  ║   Welcome to my homepage         ║
  ║                                  ║
  ║   Navigate with keyboard or      ║
  ║   click to explore               ║
  ║                                  ║
  ╚══════════════════════════════════╝
      `}</pre>
      <div style={{ color: theme.textDim, fontSize: 13, textAlign: 'center' }}>
        <div>Select a file from the left panel to preview</div>
        <div style={{ marginTop: 8, color: theme.accent2 }}>
          Type <code style={{ background: 'rgba(255,255,255,0.08)', padding: '1px 5px', borderRadius: 3 }}>:</code> or <code style={{ background: 'rgba(255,255,255,0.08)', padding: '1px 5px', borderRadius: 3 }}>/</code> to open command line
        </div>
      </div>
    </div>
  );
}

interface DirectoryPreviewProps {
  node: FSNode;
  path: string[];
  focused: boolean;
  selectedIndex: number;
  onSelect: (index: number) => void;
  onOpen: (index: number) => void;
}

function DirectoryPreview({ node, path, focused, selectedIndex, onSelect, onOpen }: DirectoryPreviewProps) {
  const entries = node.children ?? [];
  return (
    <div>
      <div style={{ color: theme.accent3, fontWeight: 600, marginBottom: 8 }}>
        {getPathString(path)}/
      </div>
      <div style={{ color: theme.textDim, marginBottom: 12 }}>
        {entries.length} items
      </div>
      {entries.map((entry, i) => {
        const isSelected = i === selectedIndex;
        return (
          <div
            key={entry.name}
            onClick={() => onSelect(i)}
            onDoubleClick={() => onOpen(i)}
            style={{
              padding: '4px 8px',
              cursor: 'pointer',
              borderRadius: 3,
              background: isSelected && focused
                ? theme.accent3 + '40'
                : isSelected
                ? 'rgba(255,255,255,0.05)'
                : 'transparent',
              borderLeft: isSelected && focused
                ? `3px solid ${theme.accent3}`
                : '3px solid transparent',
              color: entry.type === 'directory' ? theme.accent4 : theme.text,
              display: 'flex',
              alignItems: 'center',
              gap: 8,
            }}
          >
            <span style={{ fontSize: 11 }}>
              {entry.type === 'directory' ? '📁' : '📄'}
            </span>
            {entry.name}
          </div>
        );
      })}
      {focused && (
        <div style={{ marginTop: 12, fontSize: 11, color: theme.textDim }}>
          ↑↓/jk: move  Enter/l: open  Tab: switch panel
        </div>
      )}
    </div>
  );
}
