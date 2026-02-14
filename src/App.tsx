import { useState, useCallback, useEffect } from 'react';
import { theme } from './theme';
import { filesystem, resolvePath, getVisibleEntries, getPathString } from './data/filesystem';
import type { FSNode } from './data/filesystem';
import { useKeyboard } from './hooks/useKeyboard';
import type { KeyAction } from './hooks/useKeyboard';
import { useMediaQuery } from './hooks/useMediaQuery';
import { TitleBar } from './components/TitleBar';
import { FileTree } from './components/FileTree';
import { Preview } from './components/Preview';
import { CommandLine } from './components/CommandLine';
import { StatusBar } from './components/StatusBar';

type Panel = 'tree' | 'preview' | 'cmd';

function App() {
  const isMobile = useMediaQuery('(max-width: 768px)');
  const [sidebarVisible, setSidebarVisible] = useState(true);
  const [currentPath, setCurrentPath] = useState<string[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [previewSelectedIndex, setPreviewSelectedIndex] = useState(0);
  const [activePanel, setActivePanel] = useState<Panel>('tree');
  const aboutFile = filesystem.children?.find((c) => c.name === 'about.md') ?? null;
  const [previewNode, setPreviewNode] = useState<FSNode | null>(aboutFile);
  const [previewPath, setPreviewPath] = useState<string[]>([]);
  const [cmdOutput, setCmdOutput] = useState<string[]>([]);

  // Auto-hide sidebar on mobile
  useEffect(() => {
    setSidebarVisible(!isMobile);
  }, [isMobile]);

  const currentDir = resolvePath(currentPath) ?? filesystem;
  const entries = getVisibleEntries(currentDir);

  const previewEntries = previewNode?.type === 'directory' ? (previewNode.children ?? []) : [];

  const navigateTo = useCallback((path: string[]) => {
    const target = resolvePath(path);
    if (target && target.type === 'directory') {
      setCurrentPath(path);
      setSelectedIndex(0);
      setPreviewNode(target);
      setPreviewPath(path);
      setPreviewSelectedIndex(0);
    }
  }, []);

  const openEntry = useCallback((index: number) => {
    if (index === -1) {
      const newPath = currentPath.slice(0, -1);
      navigateTo(newPath);
      return;
    }
    const entry = entries[index];
    if (!entry) return;

    if (entry.type === 'directory') {
      navigateTo([...currentPath, entry.name]);
    } else {
      setPreviewNode(entry);
      setPreviewPath(currentPath);
      setActivePanel('preview');
      if (isMobile) setSidebarVisible(false);
    }
  }, [entries, currentPath, navigateTo, isMobile]);

  const openPreviewEntry = useCallback((index: number) => {
    const entry = previewEntries[index];
    if (!entry) return;

    if (entry.type === 'directory') {
      const newPath = [...previewPath, previewNode!.name, entry.name];
      // Navigate the file tree into this directory
      const parentPath = [...previewPath, previewNode!.name];
      const parentNode = resolvePath(parentPath);
      if (parentNode) {
        setCurrentPath(parentPath);
        const parentEntries = parentNode.children ?? [];
        const treeIdx = parentEntries.findIndex((e) => e.name === entry.name);
        setSelectedIndex(treeIdx >= 0 ? treeIdx : 0);
      }
      const target = resolvePath(newPath);
      if (target && target.type === 'directory') {
        setPreviewNode(target);
        setPreviewPath(newPath);
        setPreviewSelectedIndex(0);
      }
    } else {
      setPreviewNode(entry);
      setPreviewPath(previewNode?.type === 'directory' ? [...previewPath] : previewPath);
    }
  }, [previewEntries, previewPath, previewNode]);

  const selectEntry = useCallback((index: number) => {
    setSelectedIndex(index);
    const entry = entries[index];
    if (entry) {
      setPreviewNode(entry);
      setPreviewPath(currentPath);
      setPreviewSelectedIndex(0);
    }
  }, [entries, currentPath]);

  const handleCommand = useCallback((cmd: string) => {
    const parts = cmd.trim().split(/\s+/);
    const command = parts[0];
    const args = parts.slice(1);

    switch (command) {
      case 'help':
        setCmdOutput([
          'Available commands:',
          '  cd <dir>    — change directory',
          '  ls          — list files',
          '  cat <file>  — view file',
          '  clear       — clear output',
          '  whoami      — about me',
          '  help        — show this help',
        ]);
        break;
      case 'clear':
        setCmdOutput([]);
        break;
      case 'ls': {
        const dir = resolvePath(currentPath) ?? filesystem;
        const items = getVisibleEntries(dir);
        setCmdOutput(items.map((e) => (e.type === 'directory' ? `📁 ${e.name}/` : `📄 ${e.name}`)));
        break;
      }
      case 'cd': {
        const target = args[0];
        if (!target || target === '~') {
          navigateTo([]);
        } else if (target === '..') {
          navigateTo(currentPath.slice(0, -1));
        } else {
          const newPath = [...currentPath, target];
          const node = resolvePath(newPath);
          if (node && node.type === 'directory') {
            navigateTo(newPath);
            setCmdOutput([`cd ${getPathString(newPath)}`]);
          } else {
            setCmdOutput([`cd: ${target}: No such directory`]);
          }
        }
        break;
      }
      case 'cat': {
        const fileName = args[0];
        if (!fileName) {
          setCmdOutput(['cat: missing file name']);
          break;
        }
        const dir = resolvePath(currentPath) ?? filesystem;
        const file = dir.children?.find((c) => c.name === fileName);
        if (file && file.type === 'file') {
          setPreviewNode(file);
          setPreviewPath(currentPath);
          setActivePanel('preview');
          setCmdOutput([`Viewing ${fileName}`]);
        } else {
          setCmdOutput([`cat: ${fileName}: No such file`]);
        }
        break;
      }
      case 'whoami':
        setCmdOutput(['Software Engineer who loves TUI & CLI tools.']);
        break;
      default:
        setCmdOutput([`command not found: ${command}. Type 'help' for available commands.`]);
    }

    setActivePanel('tree');
  }, [currentPath, navigateTo]);

  const handleAction = useCallback((action: KeyAction) => {
    switch (action) {
      case 'up':
        if (activePanel === 'tree') {
          setSelectedIndex((prev) => {
            const next = Math.max(0, prev - 1);
            const entry = entries[next];
            if (entry) {
              setPreviewNode(entry);
              setPreviewPath(currentPath);
              setPreviewSelectedIndex(0);
            }
            return next;
          });
        } else if (activePanel === 'preview' && previewNode?.type === 'directory') {
          setPreviewSelectedIndex((prev) => Math.max(0, prev - 1));
        }
        break;
      case 'down':
        if (activePanel === 'tree') {
          setSelectedIndex((prev) => {
            const next = Math.min(entries.length - 1, prev + 1);
            const entry = entries[next];
            if (entry) {
              setPreviewNode(entry);
              setPreviewPath(currentPath);
              setPreviewSelectedIndex(0);
            }
            return next;
          });
        } else if (activePanel === 'preview' && previewNode?.type === 'directory') {
          setPreviewSelectedIndex((prev) => Math.min(previewEntries.length - 1, prev + 1));
        }
        break;
      case 'enter':
        if (activePanel === 'tree') {
          openEntry(selectedIndex);
        } else if (activePanel === 'preview' && previewNode?.type === 'directory') {
          openPreviewEntry(previewSelectedIndex);
        }
        break;
      case 'back':
        if (activePanel === 'tree' && currentPath.length > 0) {
          openEntry(-1);
        } else if (activePanel === 'preview') {
          setActivePanel('tree');
        }
        break;
      case 'tab':
        setActivePanel((prev) => {
          if (prev === 'tree') return 'preview';
          if (prev === 'preview') return 'tree';
          return 'tree';
        });
        break;
      case 'focus-cmd':
        setActivePanel('cmd');
        break;
      case 'escape':
        setActivePanel('tree');
        break;
    }
  }, [activePanel, entries, selectedIndex, currentPath, openEntry, previewNode, previewEntries, previewSelectedIndex, openPreviewEntry]);

  useKeyboard({ onAction: handleAction });

  const selectedFileName = entries[selectedIndex]?.name ?? '';

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      background: theme.bg,
      color: theme.text,
      fontFamily: "'JetBrains Mono', 'Fira Code', 'SF Mono', 'Cascadia Code', Menlo, Consolas, monospace",
    }}>
      <TitleBar
        path={getPathString(currentPath)}
        sidebarVisible={sidebarVisible}
        onToggleSidebar={() => setSidebarVisible((v) => !v)}
      />

      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {sidebarVisible && (
          <FileTree
            entries={entries}
            selectedIndex={selectedIndex}
            currentPath={currentPath}
            focused={activePanel === 'tree'}
            onSelect={selectEntry}
            onOpen={openEntry}
          />
        )}
        <Preview
          node={previewNode}
          path={previewPath}
          focused={activePanel === 'preview'}
          previewSelectedIndex={previewSelectedIndex}
          onPreviewSelect={(i) => setPreviewSelectedIndex(i)}
          onPreviewOpen={openPreviewEntry}
        />
      </div>

      <CommandLine
        focused={activePanel === 'cmd'}
        currentPath={getPathString(currentPath)}
        onCommand={handleCommand}
        onFocus={() => setActivePanel('cmd')}
        onBlur={() => {}}
        output={cmdOutput}
      />

      <StatusBar
        activePanel={activePanel}
        fileCount={entries.length}
        selectedFile={selectedFileName}
      />
    </div>
  );
}

export default App;
