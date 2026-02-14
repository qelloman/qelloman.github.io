import { theme } from '../theme';

interface TitleBarProps {
  path: string;
  sidebarVisible: boolean;
  onToggleSidebar: () => void;
}

export function TitleBar({ path, sidebarVisible, onToggleSidebar }: TitleBarProps) {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      padding: '6px 12px',
      background: theme.panel,
      borderBottom: `1px solid ${theme.border}`,
    }}>
      <button
        onClick={onToggleSidebar}
        aria-label="Toggle sidebar"
        style={{
          background: sidebarVisible ? 'transparent' : theme.accent3,
          border: `1px solid ${sidebarVisible ? theme.accent3 : theme.accent3}`,
          color: sidebarVisible ? theme.accent3 : theme.textBright,
          fontSize: 14,
          padding: '3px 7px',
          borderRadius: 4,
          cursor: 'pointer',
          marginRight: 12,
          fontFamily: 'inherit',
          lineHeight: 1,
          display: 'flex',
          flexDirection: 'column',
          gap: 2,
          alignItems: 'center',
          justifyContent: 'center',
          width: 28,
          height: 24,
        }}
      >
        <span style={{ width: 14, height: 2, background: sidebarVisible ? theme.accent3 : theme.textBright, borderRadius: 1 }} />
        <span style={{ width: 14, height: 2, background: sidebarVisible ? theme.accent3 : theme.textBright, borderRadius: 1 }} />
        <span style={{ width: 14, height: 2, background: sidebarVisible ? theme.accent3 : theme.textBright, borderRadius: 1 }} />
      </button>
      <span style={{ color: theme.textDim, fontSize: 13 }}>
        doyunkim@homepage: {path}
      </span>
    </div>
  );
}
