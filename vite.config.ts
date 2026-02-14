import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'markdown-loader',
      transform(src, id) {
        if (id.endsWith('.md')) {
          return {
            code: `export default ${JSON.stringify(src)}`,
            map: null,
          }
        }
      },
    },
  ],
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
      '@content': resolve(__dirname, './content'),
    },
  },
})
