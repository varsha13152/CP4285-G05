// usage: node render.js preview 2.0 4.5 8   |   node render.js all [fps] [scale]
const puppeteer = require('puppeteer-core');
const path = require('path');
const fs = require('fs');
(async () => {
  const mode = process.argv[2] || 'preview';
  const fps = mode === 'all' ? parseFloat(process.argv[3] || '20') : 0;
  const scale = mode === 'all' ? parseFloat(process.argv[4] || '1.5') : 1;
  const b = await puppeteer.launch({
    executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    headless: true, args: ['--allow-file-access-from-files', '--force-color-profile=srgb'],
  });
  const p = await b.newPage();
  await p.setViewport({ width: 1280, height: 720, deviceScaleFactor: scale });
  await p.goto('file:///' + path.resolve(__dirname, 'scene.html').replace(/\\/g, '/'), { waitUntil: 'networkidle0' });
  await p.evaluate(async () => { await document.fonts.ready; });
  const fam = await p.evaluate(() => document.fonts.check('800 62px Poppins'));
  console.log('poppins loaded:', fam);
  if (mode === 'all') {
    const dir = path.join(__dirname, 'frames');
    fs.rmSync(dir, { recursive: true, force: true }); fs.mkdirSync(dir);
    const T = await p.evaluate(() => window.T);
    const n = Math.round(T * fps);
    for (let i = 0; i < n; i++) {
      await p.evaluate(t => window.render(t), i / fps);
      await p.screenshot({ path: path.join(dir, `f${String(i).padStart(4, '0')}.png`) });
    }
    console.log('frames', n);
  } else {
    for (const t of process.argv.slice(3)) {
      await p.evaluate(t => window.render(t), parseFloat(t));
      await p.screenshot({ path: path.join(__dirname, `prev_${t}.png`) });
    }
  }
  await b.close();
})();
