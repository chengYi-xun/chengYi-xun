const { escapeHTML } = require('hexo-util');
const shiki = require('shiki');

const config = hexo.config.shiki || {};
const enabled = hexo.config.syntax_highlighter === 'shiki';

const lightTheme = (config.themes && config.themes.light) || config.theme || 'one-light';
const darkTheme = (config.themes && config.themes.dark) || null;
const allThemes = darkTheme ? [lightTheme, darkTheme] : [lightTheme];

const highlighter = await shiki.createHighlighter({
  themes: allThemes,
  langs: Array.isArray(config.langs) ? config.langs : Object.keys(shiki.bundledLanguages),
  langAlias: config.lang_alias || undefined,
});

const missingLanguages = new Set();

hexo.extend.highlight.register(
  'shiki',
  (code, options) => {
    try {
      let themeOpts;
      if (darkTheme) {
        themeOpts = {
          themes: { light: lightTheme, dark: darkTheme },
          defaultColor: 'light',
        };
      } else {
        themeOpts = { theme: lightTheme };
      }

      const highlighted = highlighter.codeToHtml(code, {
        ...themeOpts,
        lang: options.lang || 'plain',
      });

      const langLabel = options.lang
        ? `<div class="shiki-lang-label">${escapeHTML(options.lang)}</div>`
        : '';

      return `<div class="shiki-container">${langLabel}<button class="shiki-copy-btn" onclick="navigator.clipboard.writeText(this.parentElement.querySelector('code').textContent)">Copy</button>${highlighted}</div>`;
    } catch (error) {
      const m = error.message.match(/^Language `(.+?)` not found/);
      if (m) {
        missingLanguages.add(m[1]);
      } else {
        hexo.log.warn('shiki highlight error:', error.message);
      }
      return `<pre><code>${escapeHTML(code)}</code></pre>`;
    }
  },
);

if (enabled) {
  hexo.extend.filter.register('before_exit', () => {
    if (missingLanguages.size) {
      hexo.log.warn('shiki missing languages:', Array.from(missingLanguages).sort().join(', '));
    }
  });

  hexo.extend.injector.register('head_end', `
    <style>
      .shiki-container {
        position: relative;
        margin: 1rem 0;
        border-radius: 8px;
        overflow: hidden;
      }
      .shiki-container pre.shiki {
        margin: 0;
        padding: 1.2rem 1rem;
        overflow-x: auto;
        border-radius: 8px;
        font-size: 0.9rem;
        line-height: 1.6;
      }
      .shiki-container pre.shiki code {
        counter-reset: step;
        counter-increment: step 0;
      }
      .shiki-container pre.shiki code .line::before {
        content: counter(step);
        counter-increment: step;
        width: 1.5rem;
        margin-right: 1rem;
        display: inline-block;
        text-align: right;
        color: rgba(115,138,148,.4);
        user-select: none;
      }
      .shiki-lang-label {
        position: absolute;
        top: 6px;
        right: 70px;
        font-size: 0.75rem;
        color: rgba(150,150,150,.8);
        text-transform: uppercase;
        pointer-events: none;
        z-index: 1;
      }
      .shiki-copy-btn {
        position: absolute;
        top: 6px;
        right: 8px;
        padding: 2px 10px;
        font-size: 0.75rem;
        cursor: pointer;
        border: 1px solid rgba(150,150,150,.3);
        border-radius: 4px;
        background: rgba(255,255,255,.1);
        color: rgba(150,150,150,.8);
        opacity: 0;
        transition: opacity .2s;
        z-index: 1;
      }
      .shiki-container:hover .shiki-copy-btn { opacity: 1; }
      .shiki-copy-btn:hover { background: rgba(255,255,255,.2); color: #fff; }

      ${darkTheme ? `
        [data-theme="dark"] .shiki,
        [data-theme="dark"] .shiki span {
          color: var(--shiki-dark) !important;
          background-color: var(--shiki-dark-bg) !important;
          font-style: var(--shiki-dark-font-style) !important;
          font-weight: var(--shiki-dark-font-weight) !important;
          text-decoration: var(--shiki-dark-text-decoration) !important;
        }
        @media (prefers-color-scheme: dark) {
          .shiki,
          .shiki span {
            color: var(--shiki-dark) !important;
            background-color: var(--shiki-dark-bg) !important;
            font-style: var(--shiki-dark-font-style) !important;
            font-weight: var(--shiki-dark-font-weight) !important;
            text-decoration: var(--shiki-dark-text-decoration) !important;
          }
        }
      ` : ''}
    </style>
  `, 'post');
}
