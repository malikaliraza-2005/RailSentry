/** Colours come from CSS variables, so one attribute switches dark/light. */
const token = (name) => `rgb(var(--${name}) / <alpha-value>)`;

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: token("bg"),
        surface: token("surface"),
        "surface-2": token("surface-2"),
        border: token("border"),
        text: token("text"),
        muted: token("text-muted"),
        accent: token("accent"),
        onaccent: token("on-accent"),
        success: token("success"),
        warning: token("warning"),
        danger: token("danger"),
        path: token("path"),
        board: "#111210",
        flap: "#ffcd28",
      },
      fontFamily: {
        sans: ["Barlow", "Arial", "sans-serif"],
        display: ["Barlow Condensed", "Arial Narrow", "sans-serif"],
        mono: ["Barlow", "Arial", "sans-serif"],
      },
      // Printed signage has square corners.
      borderRadius: { sm: "0", DEFAULT: "0", md: "0", lg: "0", xl: "0", card: "0" },
    },
  },
  plugins: [],
};
