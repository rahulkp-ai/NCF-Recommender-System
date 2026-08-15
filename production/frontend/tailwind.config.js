/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx}",
    "./src/components/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        netflix: {
          red:    "#E50914",
          dark:   "#141414",
          gray:   "#2F2F2F",
          light:  "#B3B3B3",
        },
      },
      fontFamily: {
        sans: ["Netflix Sans", "Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
      },
    },
  },
  plugins: [],
};
