import { useState, useRef, useEffect } from 'react';
import { theme } from '../theme';

interface CommandLineProps {
  focused: boolean;
  currentPath: string;
  onCommand: (cmd: string) => void;
  onFocus: () => void;
  onBlur: () => void;
  output: string[];
}

export function CommandLine({ focused, currentPath, onCommand, onFocus, onBlur, output }: CommandLineProps) {
  const [input, setInput] = useState('');
  const [history, setHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (focused && inputRef.current) {
      inputRef.current.focus();
    }
  }, [focused]);

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter') {
      e.preventDefault();
      e.stopPropagation();
      const cmd = input.trim();
      if (cmd) {
        setHistory((prev) => [cmd, ...prev]);
        onCommand(cmd);
      }
      setInput('');
      setHistoryIndex(-1);
      return;
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      const next = Math.min(historyIndex + 1, history.length - 1);
      setHistoryIndex(next);
      if (history[next]) setInput(history[next]);
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      const next = historyIndex - 1;
      if (next < 0) {
        setHistoryIndex(-1);
        setInput('');
      } else {
        setHistoryIndex(next);
        setInput(history[next]);
      }
    }
  }

  return (
    <div style={{
      borderTop: `1px solid ${theme.border}`,
      background: theme.panel,
    }}>
      {output.length > 0 && (
        <div style={{
          padding: '4px 12px',
          fontSize: 12,
          color: theme.textDim,
          maxHeight: 80,
          overflowY: 'auto',
          borderBottom: `1px solid ${theme.border}`,
        }}>
          {output.map((line, i) => (
            <div key={i}>{line}</div>
          ))}
        </div>
      )}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        padding: '4px 12px',
        gap: 8,
      }}>
        <span style={{ color: theme.accent2, fontSize: 13, flexShrink: 0 }}>
          {currentPath} $
        </span>
        <input
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onFocus={onFocus}
          onBlur={onBlur}
          onKeyDown={handleKeyDown}
          placeholder={focused ? 'Type a command...' : 'Press : or / to focus'}
          style={{
            flex: 1,
            background: 'transparent',
            border: 'none',
            outline: 'none',
            color: theme.text,
            fontSize: 13,
            fontFamily: 'inherit',
          }}
        />
      </div>
    </div>
  );
}
