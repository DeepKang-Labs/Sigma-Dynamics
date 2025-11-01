// scripts/capture_visual.js
const fs = require("fs");
const path = require("path");
const puppeteer = require("puppeteer");

(async () => {
  const outDir = path.join(process.cwd(), "docs", "img");
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

  const url = process.env.CAPTURE_URL || "http://localhost:3000/visuals";
  const outPath = path.join(outDir, "sigma_influence_flow.png");

  const browser = await puppeteer.launch({
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 2 });
  await page.goto(url, { waitUntil: "networkidle2", timeout: 120000 });

  // Optionnel : attendre que l’animation Σ apparaisse
  await page.waitForSelector("span", { timeout: 60000 });

  await page.screenshot({ path: outPath, fullPage: false });
  await browser.close();

  console.log(`✅ Image générée: ${outPath}`);
})();
