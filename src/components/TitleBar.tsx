import { theme } from '../theme';

interface TitleBarProps {
  path: string;
}

export function TitleBar({ path }: TitleBarProps) {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      padding: '6px 12px',
      background: theme.panel,
      borderBottom: `1px solid ${theme.border}`,
    }}>
      <div style={{ display: 'flex', gap: 6, marginRight: 16 }}>
        <span style={{ width: 12, height: 12, borderRadius: '50%', background: theme.accent1, display: 'inline-block' }} />
        <span style={{ width: 12, height: 12, borderRadius: '50%', background: theme.accent2, display: 'inline-block' }} />
        <span style={{ width: 12, height: 12, borderRadius: '50%', background: '#27ae60', display: 'inline-block' }} />
      </div>
      <span style={{ color: theme.textDim, fontSize: 13 }}>
        doyunkim@homepage: {path}
      </span>
    </div>
  );
}
