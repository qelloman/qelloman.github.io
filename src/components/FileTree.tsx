import { theme } from '../theme';
import type { FSNode } from '../data/filesystem';

interface FileTreeProps {
  entries: FSNode[];
  selectedIndex: number;
  currentPath: string[];
  focused: boolean;
  onSelect: (index: number) => void;
  onOpen: (index: number) => void;
}

export function FileTree({ entries, selectedIndex, currentPath, focused, onSelect, onOpen }: FileTreeProps) {
  return (
    <div style={{
      width: 260,
      minWidth: 200,
      borderRight: `1px solid ${theme.border}`,
      display: 'flex',
      flexDirection: 'column',
      background: theme.panel,
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
        Files {currentPath.length > 0 && `(${currentPath.join('/')})`}
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: '4px 0' }}>
        {currentPath.length > 0 && (
          <div
            style={{
              padding: '4px 12px',
              cursor: 'pointer',
              color: theme.textDim,
              fontSize: 13,
            }}
            onClick={() => onOpen(-1)}
          >
            ..
          </div>
        )}
        {entries.map((entry, i) => {
          const isSelected = i === selectedIndex;
          const isDir = entry.type === 'directory';
          return (
            <div
              key={entry.name}
              onClick={() => onSelect(i)}
              onDoubleClick={() => onOpen(i)}
              style={{
                padding: '4px 12px',
                cursor: 'pointer',
                background: isSelected && focused
                  ? theme.accent3 + '40'
                  : isSelected
                  ? 'rgba(255,255,255,0.05)'
                  : 'transparent',
                borderLeft: isSelected && focused
                  ? `3px solid ${theme.accent3}`
                  : '3px solid transparent',
                color: isDir ? theme.accent4 : theme.text,
                fontSize: 13,
                display: 'flex',
                alignItems: 'center',
                gap: 8,
              }}
            >
              <span style={{ fontSize: 11, width: 16, textAlign: 'center' }}>
                {isDir ? '📁' : '📄'}
              </span>
              <span>{entry.name}</span>
            </div>
          );
        })}
      </div>

      <div style={{
        padding: '6px 12px',
        fontSize: 11,
        color: theme.textDim,
        borderTop: `1px solid ${theme.border}`,
      }}>
        ↑↓/jk: move  Enter/l: open  h: back
      </div>
    </div>
  );
}
