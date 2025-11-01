/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        sigma: {
          black: "#050505",
          cyan: "#00FFFF",
          indigo: "#3B82F6",
          violet: "#8B5CF6",
          rose: "#E11D48",
          gold: "#FACC15",
          emerald: "#10B981",
        },
      },
      fontFamily: {
        sigma: ["Orbitron", "sans-serif"],
      },
      boxShadow: {
        "sigma-glow": "0 0 40px rgba(0,255,255,0.5)",
      },
      backgroundImage: {
        "sigma-gradient":
          "linear-gradient(135deg, rgba(0,255,255,0.3) 0%, rgba(139,92,246,0.3) 100%)",
      },
    },
  },
  plugins: [],
};
