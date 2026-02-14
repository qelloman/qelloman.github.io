import { useEffect } from 'react';

export type KeyAction =
  | 'up'
  | 'down'
  | 'enter'
  | 'back'
  | 'tab'
  | 'focus-cmd'
  | 'escape';

interface UseKeyboardOptions {
  onAction: (action: KeyAction) => void;
  enabled?: boolean;
}

export function useKeyboard({ onAction, enabled = true }: UseKeyboardOptions) {
  useEffect(() => {
    if (!enabled) return;

    function handleKeyDown(e: KeyboardEvent) {
      const target = e.target as HTMLElement;
      const isInput = target.tagName === 'INPUT' || target.tagName === 'TEXTAREA';

      if (isInput) {
        if (e.key === 'Escape') {
          e.preventDefault();
          onAction('escape');
        }
        // Enter in input is handled by the input component itself
        return;
      }

      switch (e.key) {
        case 'ArrowUp':
        case 'k':
          e.preventDefault();
          onAction('up');
          break;
        case 'ArrowDown':
        case 'j':
          e.preventDefault();
          onAction('down');
          break;
        case 'Enter':
        case 'l':
        case 'ArrowRight':
          e.preventDefault();
          onAction('enter');
          break;
        case 'h':
        case 'ArrowLeft':
        case 'Backspace':
          e.preventDefault();
          onAction('back');
          break;
        case 'Tab':
          e.preventDefault();
          onAction('tab');
          break;
        case ':':
        case '/':
          e.preventDefault();
          onAction('focus-cmd');
          break;
        case 'Escape':
          e.preventDefault();
          onAction('escape');
          break;
      }
    }

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onAction, enabled]);
}
