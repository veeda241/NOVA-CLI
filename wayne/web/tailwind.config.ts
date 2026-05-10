import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}", "./components/**/*.{js,ts,jsx,tsx,mdx}", "./hooks/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        background: "#060c14",
        surface: "#0a1628",
        panel: "#0d1f38",
        cyan: "#00d4ff",
        amber: "#ffaa00",
        success: "#00ff88",
        danger: "#ff4455"
      },
      fontFamily: {
        body: ["var(--font-share-tech)", "monospace"],
        heading: ["var(--font-orbitron)", "monospace"]
      },
      animation: {
        fade: "fade 220ms ease-out",
        pulseMic: "pulseMic 1.2s ease-in-out infinite",
        bounceDot: "bounceDot 900ms infinite"
      },
      keyframes: {
        fade: {
          "0%": { opacity: "0", transform: "translateY(4px)" },
          "100%": { opacity: "1", transform: "translateY(0)" }
        },
        pulseMic: {
          "0%, 100%": { boxShadow: "0 0 0 0 rgba(255,68,85,.5)" },
          "50%": { boxShadow: "0 0 0 8px rgba(255,68,85,0)" }
        },
        bounceDot: {
          "0%, 80%, 100%": { transform: "translateY(0)", opacity: ".45" },
          "40%": { transform: "translateY(-5px)", opacity: "1" }
        }
      }
    }
  },
  plugins: []
};

export default config;
