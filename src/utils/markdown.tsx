import React from 'react';
import { theme } from '../theme';

interface MarkdownLine {
  type: 'h1' | 'h2' | 'h3' | 'blockquote' | 'ul' | 'ol' | 'hr' | 'table-header' | 'table-sep' | 'table-row' | 'code-block-toggle' | 'paragraph';
  raw: string;
}

function classifyLine(line: string): MarkdownLine['type'] {
  const trimmed = line.trimStart();
  if (trimmed.startsWith('# ')) return 'h1';
  if (trimmed.startsWith('## ')) return 'h2';
  if (trimmed.startsWith('### ')) return 'h3';
  if (trimmed.startsWith('> ')) return 'blockquote';
  if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) return 'ul';
  if (/^\d+\.\s/.test(trimmed)) return 'ol';
  if (trimmed.startsWith('---') || trimmed.startsWith('***')) return 'hr';
  if (trimmed.startsWith('```')) return 'code-block-toggle';
  if (trimmed.startsWith('|') && trimmed.includes('---')) return 'table-sep';
  if (trimmed.startsWith('|')) return 'table-row';
  return 'paragraph';
}

function renderInline(text: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = [];
  const regex = /(!\[([^\]]*)\]\(([^)]+)\))|(\*\*(.+?)\*\*)|(`(.+?)`)|(\*(.+?)\*)|(\[(.+?)\]\((.+?)\))/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  let key = 0;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(text.slice(lastIndex, match.index));
    }
    if (match[1]) {
      // ![alt](url) image
      nodes.push(
        <img key={key++} src={match[3]} alt={match[2]}
          style={{
            maxWidth: '100%',
            maxHeight: 400,
            borderRadius: 4,
            border: `1px solid ${theme.border}`,
            margin: '8px 0',
            display: 'block',
          }}
        />
      );
    } else if (match[4]) {
      nodes.push(<strong key={key++} style={{ color: theme.textBright, fontWeight: 700 }}>{match[5]}</strong>);
    } else if (match[6]) {
      nodes.push(
        <code key={key++} style={{
          background: 'rgba(255,255,255,0.08)',
          padding: '1px 5px',
          borderRadius: 3,
          color: theme.accent2,
          fontSize: '0.9em',
        }}>{match[7]}</code>
      );
    } else if (match[8]) {
      nodes.push(<em key={key++} style={{ color: theme.textDim, fontStyle: 'italic' }}>{match[9]}</em>);
    } else if (match[10]) {
      nodes.push(
        <a key={key++} href={match[12]} target="_blank" rel="noopener noreferrer"
          style={{ color: theme.accent4, textDecoration: 'underline' }}>
          {match[11]}
        </a>
      );
    }
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < text.length) {
    nodes.push(text.slice(lastIndex));
  }
  return nodes;
}

export function renderMarkdown(markdown: string): React.ReactElement {
  const lines = markdown.split('\n');
  const elements: React.ReactElement[] = [];
  let inCodeBlock = false;
  let codeLines: string[] = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const type = classifyLine(line);

    if (type === 'code-block-toggle') {
      if (inCodeBlock) {
        elements.push(
          <pre key={i} style={{
            background: 'rgba(0,0,0,0.3)',
            padding: '8px 12px',
            borderRadius: 4,
            margin: '4px 0',
            fontSize: '0.9em',
            overflowX: 'auto',
            color: theme.text,
          }}>
            {codeLines.join('\n')}
          </pre>
        );
        codeLines = [];
      }
      inCodeBlock = !inCodeBlock;
      continue;
    }

    if (inCodeBlock) {
      codeLines.push(line);
      continue;
    }

    const trimmed = line.trimStart();

    switch (type) {
      case 'h1':
        elements.push(
          <div key={i} style={{ color: theme.accent1, fontSize: '1.4em', fontWeight: 700, margin: '8px 0 4px' }}>
            {renderInline(trimmed.slice(2))}
          </div>
        );
        break;
      case 'h2':
        elements.push(
          <div key={i} style={{ color: theme.accent3, fontSize: '1.15em', fontWeight: 600, margin: '10px 0 2px' }}>
            {renderInline(trimmed.slice(3))}
          </div>
        );
        break;
      case 'h3':
        elements.push(
          <div key={i} style={{ color: theme.accent4, fontSize: '1.05em', fontWeight: 600, margin: '6px 0 2px' }}>
            {renderInline(trimmed.slice(4))}
          </div>
        );
        break;
      case 'blockquote':
        elements.push(
          <div key={i} style={{
            borderLeft: `3px solid ${theme.accent3}`,
            paddingLeft: 12,
            color: theme.textDim,
            fontStyle: 'italic',
            margin: '4px 0',
          }}>
            {renderInline(trimmed.slice(2))}
          </div>
        );
        break;
      case 'ul':
        elements.push(
          <div key={i} style={{ paddingLeft: 16, margin: '2px 0' }}>
            <span style={{ color: theme.accent2 }}>  </span>
            {renderInline(trimmed.slice(2))}
          </div>
        );
        break;
      case 'ol': {
        const olMatch = trimmed.match(/^(\d+)\.\s(.*)$/);
        if (olMatch) {
          elements.push(
            <div key={i} style={{ paddingLeft: 16, margin: '2px 0' }}>
              <span style={{ color: theme.accent2 }}>{olMatch[1]}. </span>
              {renderInline(olMatch[2])}
            </div>
          );
        }
        break;
      }
      case 'hr':
        elements.push(
          <hr key={i} style={{ border: 'none', borderTop: `1px solid ${theme.border}`, margin: '8px 0' }} />
        );
        break;
      case 'table-sep':
        break;
      case 'table-row': {
        const cells = trimmed.split('|').filter((c) => c.trim() !== '');
        elements.push(
          <div key={i} style={{ display: 'flex', gap: 16, margin: '2px 0', paddingLeft: 4 }}>
            {cells.map((cell, ci) => (
              <span key={ci} style={{ minWidth: 80 }}>{renderInline(cell.trim())}</span>
            ))}
          </div>
        );
        break;
      }
      case 'paragraph':
        if (trimmed === '') {
          elements.push(<div key={i} style={{ height: 6 }} />);
        } else {
          elements.push(
            <div key={i} style={{ margin: '2px 0' }}>
              {renderInline(trimmed)}
            </div>
          );
        }
        break;
    }
  }

  return <div>{elements}</div>;
}
