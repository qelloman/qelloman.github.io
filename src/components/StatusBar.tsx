import { theme } from '../theme';

interface StatusBarProps {
  activePanel: 'tree' | 'preview' | 'cmd';
  fileCount: number;
  selectedFile: string;
}

export function StatusBar({ activePanel, fileCount, selectedFile }: StatusBarProps) {
  return (
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '3px 12px',
      background: theme.accent3,
      fontSize: 12,
      color: theme.textBright,
    }}>
      <div style={{ display: 'flex', gap: 16 }}>
        <span style={{ fontWeight: activePanel === 'tree' ? 700 : 400 }}>
          [1] Files
        </span>
        <span style={{ fontWeight: activePanel === 'preview' ? 700 : 400 }}>
          [2] Preview
        </span>
        <span style={{ fontWeight: activePanel === 'cmd' ? 700 : 400 }}>
          [:] Command
        </span>
      </div>
      <div style={{ display: 'flex', gap: 16 }}>
        <span>{selectedFile || 'No file selected'}</span>
        <span>{fileCount} items</span>
        <span>Tab: switch panel</span>
      </div>
    </div>
  );
}
