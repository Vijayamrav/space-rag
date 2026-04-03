/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        surface:  '#212121',
        sidebar:  '#171717',
        input:    '#2f2f2f',
        border:   '#3f3f3f',
        accent:   '#10a37f',
      },
    },
  },
  plugins: [],
}
