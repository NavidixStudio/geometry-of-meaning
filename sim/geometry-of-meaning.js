/* =====================================================================
   NAVIDIX — THE GEOMETRY OF MEANING (EXPERIENCE 06)
   Interactive 3D Semantic Space, Vector Algebra, Self-Attention,
   Softmax Temperature & 8D-to-3D Projection Observatory.

   WHAT IS ACTUALLY COMPUTED HERE (NO FAKED ANIMATIONS):
     1. Every one of the 36 Persian words + the polysemous probe "شیر"
        carries a real unit-normalized 8-dimensional vector v in R^8.
     2. Every cosine similarity cos(theta) = u . v and angle acos(u . v)
        shown in the HUD and Inspector is computed live in R^8.
     3. The 3D positions are an exact linear orthonormal projection
        p = R * P(phi) * v where P(phi) in R^{3x8} satisfies P P^T = I_3
        for every slice angle phi (exact SO(8) Givens slice rotation).
        Because P is linear, every 3D analogy parallelogram
        P(a - b + c) = P(a) - P(b) + P(c) holds geometrically.
     4. Chapter 03 computes real Scaled Dot-Product Attention weights
        alpha_j = softmax(s_j / tau) and moves "شیر" along a smooth
        residual update trajectory across Transformer layers L = 0..12.
     5. Chapter 04 computes exact Softmax probabilities
        P_i(T) = exp(z_i / T) / sum_j exp(z_j / T), real Top-P nucleus
        truncation, live Shannon entropy H = -sum P_i log2(P_i),
        perplexity 2^H, and real weighted random token sampling.
   ===================================================================== */
(function (global) {
  'use strict';

  var THREE_SRC = './assets/three.min.js';
  var threePromise = null;

  function loadThree() {
    if (global.THREE) return Promise.resolve(global.THREE);
    if (threePromise) return threePromise;
    threePromise = new Promise(function (resolve, reject) {
      var s = document.createElement('script');
      s.src = THREE_SRC;
      s.async = true;
      s.onload = function () {
        global.THREE ? resolve(global.THREE) : reject(new Error('Three.js loaded without global THREE'));
      };
      s.onerror = function () { reject(new Error('Failed to load Three.js')); };
      document.head.appendChild(s);
    });
    return threePromise;
  }

  function hasWebGL() {
    try {
      var c = document.createElement('canvas');
      return !!(global.WebGLRenderingContext && (c.getContext('webgl') || c.getContext('experimental-webgl')));
    } catch (_) { return false; }
  }

  var FA_DIGITS = '۰۱۲۳۴۵۶۷۸۹';
  function fa(v) {
    return String(v).replace(/\d/g, function (d) { return FA_DIGITS[d]; }).replace(/\./g, '٫');
  }

  /* =====================================================================
     1. 8-DIMENSIONAL SEMANTIC LEXICON & LINEAR ALGEBRA
     Dimensions:
       0: جاندار / انسان    1: جایگاه / مرکزیت   2: جنسیت (مرد+/زن-)
       3: جغرافیا / مکان   4: خوراک / آشپزخانه   5: حیات‌وحش / طبیعت
       6: آب / ابزار       7: علم / منطق / هوش
     ===================================================================== */
  var DIM_NAMES = [
    'جاندار / انسان', 'جایگاه / قدرت', 'جنسیت', 'جغرافیا',
    'خوراک / گرما', 'حیات‌وحش', 'آب / ابزار', 'علم / منطق'
  ];

  var CLUSTERS = {
    royalty:   { name: 'سلطنت و انسان',   hex: '#F5B754', color: 0xF5B754 },
    geography: { name: 'جغرافیا و پایتخت', hex: '#8FDBFF', color: 0x8FDBFF },
    wildlife:  { name: 'حیات‌وحش و طبیعت', hex: '#FF6A5A', color: 0xFF6A5A },
    food:      { name: 'خوراک و صبحانه',   hex: '#3EE0A8', color: 0x3EE0A8 },
    water:     { name: 'آب و ابزار',       hex: '#5294FF', color: 0x5294FF },
    science:   { name: 'علم و هوش مصنوعی', hex: '#A78BFA', color: 0xA78BFA }
  };

  function vNorm(a) {
    var s = 0;
    for (var i = 0; i < a.length; i++) s += a[i] * a[i];
    var m = Math.sqrt(s) || 1;
    var out = new Array(a.length);
    for (var j = 0; j < a.length; j++) out[j] = a[j] / m;
    return out;
  }
  function vDot(a, b) {
    var s = 0;
    for (var i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }
  function vSub(a, b) {
    var out = new Array(a.length);
    for (var i = 0; i < a.length; i++) out[i] = a[i] - b[i];
    return out;
  }
  function vAdd(a, b) {
    var out = new Array(a.length);
    for (var i = 0; i < a.length; i++) out[i] = a[i] + b[i];
    return out;
  }
  function vScale(a, k) {
    var out = new Array(a.length);
    for (var i = 0; i < a.length; i++) out[i] = a[i] * k;
    return out;
  }
  function vLerp(a, b, t) {
    var out = new Array(a.length);
    for (var i = 0; i < a.length; i++) out[i] = a[i] + (b[i] - a[i]) * t;
    return out;
  }
  function cosSim(a, b) {
    var d = vDot(a, b);
    return Math.max(-1, Math.min(1, d));
  }
  function angleDeg(a, b) {
    return Math.acos(cosSim(a, b)) * (180 / Math.PI);
  }

  /* 36 Persian lexicon entries with carefully calibrated 8D latent vectors */
  var RAW_WORDS = [
    /* Royalty & Human */
    { id: 'king',      fa: 'پادشاه',   c: 'royalty',   v: [0.66, 0.94,  0.64, 0.05, 0.00, 0.02, 0.00, 0.02] },
    { id: 'queen',     fa: 'ملکه',     c: 'royalty',   v: [0.66, 0.94, -0.64, 0.05, 0.00, 0.02, 0.00, 0.02] },
    { id: 'man',       fa: 'مرد',      c: 'royalty',   v: [0.88, 0.04,  0.72, 0.00, 0.00, 0.04, 0.00, 0.02] },
    { id: 'woman',     fa: 'زن',       c: 'royalty',   v: [0.88, 0.04, -0.72, 0.00, 0.00, 0.04, 0.00, 0.02] },
    { id: 'prince',    fa: 'شاهزاده',  c: 'royalty',   v: [0.74, 0.56,  0.64, 0.02, 0.00, 0.02, 0.00, 0.02] },
    { id: 'princess',  fa: 'شاهدخت',   c: 'royalty',   v: [0.74, 0.56, -0.64, 0.02, 0.00, 0.02, 0.00, 0.02] },

    /* Geography & Capitals (same +status shift from country to capital) */
    { id: 'iran',      fa: 'ایران',    c: 'geography', v: [0.00, 0.08,  0.00, 0.96, 0.16, 0.12, 0.00, 0.08] },
    { id: 'tehran',    fa: 'تهران',    c: 'geography', v: [0.00, 0.74,  0.00, 0.96, 0.16, 0.12, 0.00, 0.08] },
    { id: 'france',    fa: 'فرانسه',   c: 'geography', v: [0.00, 0.08, -0.30, 0.90, 0.24, 0.00, 0.00, 0.22] },
    { id: 'paris',     fa: 'پاریس',    c: 'geography', v: [0.00, 0.74, -0.30, 0.90, 0.24, 0.00, 0.00, 0.22] },
    { id: 'japan',     fa: 'ژاپن',     c: 'geography', v: [0.00, 0.08,  0.30, 0.88, 0.06, 0.00, 0.18, 0.32] },
    { id: 'tokyo',     fa: 'توکیو',    c: 'geography', v: [0.00, 0.74,  0.30, 0.88, 0.06, 0.00, 0.18, 0.32] },

    /* Wildlife & Savanna (World 1 of شیر) */
    { id: 'tiger',     fa: 'ببر',      c: 'wildlife',  v: [0.42, 0.24,  0.12, 0.00, 0.02, 0.96, 0.00, 0.00] },
    { id: 'leopard',   fa: 'پلنگ',     c: 'wildlife',  v: [0.38, 0.14,  0.06, 0.00, 0.00, 0.94, 0.00, 0.00] },
    { id: 'jungle',    fa: 'جنگل',     c: 'wildlife',  v: [0.08, 0.00,  0.00, 0.34, 0.04, 0.92, 0.14, 0.00] },
    { id: 'roar',      fa: 'غرش',      c: 'wildlife',  v: [0.22, 0.32,  0.22, 0.00, 0.00, 0.90, 0.00, 0.00] },
    { id: 'hunt',      fa: 'شکار',     c: 'wildlife',  v: [0.28, 0.16,  0.18, 0.06, 0.14, 0.88, 0.00, 0.00] },
    { id: 'deer',      fa: 'آهو',      c: 'wildlife',  v: [0.48, 0.00, -0.18, 0.08, 0.04, 0.86, 0.00, 0.00] },

    /* Food & Breakfast (World 2 of شیر) */
    { id: 'coffee',    fa: 'قهوه',     c: 'food',      v: [0.00, 0.02,  0.00, 0.06, 0.96, 0.00, 0.24, 0.04] },
    { id: 'bread',     fa: 'نان',      c: 'food',      v: [0.00, 0.00,  0.00, 0.04, 0.94, 0.08, 0.02, 0.00] },
    { id: 'honey',     fa: 'عسل',      c: 'food',      v: [0.04, 0.06,  0.00, 0.02, 0.92, 0.24, 0.10, 0.00] },
    { id: 'breakfast', fa: 'صبحانه',   c: 'food',      v: [0.14, 0.00,  0.00, 0.00, 0.95, 0.00, 0.08, 0.00] },
    { id: 'cup',       fa: 'فنجان',    c: 'food',      v: [0.00, 0.00, -0.06, 0.00, 0.86, 0.00, 0.36, 0.00] },
    { id: 'warm',      fa: 'داغ',      c: 'food',      v: [0.00, 0.06,  0.06, 0.00, 0.84, 0.08, 0.22, 0.14] },

    /* Water & Plumbing (World 3 of شیر) */
    { id: 'pipe',      fa: 'لوله',     c: 'water',     v: [0.00, 0.00,  0.00, 0.04, 0.02, 0.00, 0.96, 0.18] },
    { id: 'drop',      fa: 'قطره',     c: 'water',     v: [0.00, 0.00,  0.00, 0.02, 0.26, 0.16, 0.92, 0.06] },
    { id: 'wrench',    fa: 'آچار',     c: 'water',     v: [0.00, 0.06,  0.14, 0.00, 0.00, 0.00, 0.90, 0.34] },
    { id: 'basin',     fa: 'حوض',      c: 'water',     v: [0.00, 0.00,  0.00, 0.24, 0.06, 0.10, 0.90, 0.00] },
    { id: 'pressure',  fa: 'فشار',     c: 'water',     v: [0.00, 0.14,  0.06, 0.00, 0.00, 0.00, 0.84, 0.44] },
    { id: 'drip',      fa: 'چکه',      c: 'water',     v: [0.00, 0.00,  0.00, 0.00, 0.12, 0.06, 0.94, 0.06] },

    /* Science & AI */
    { id: 'vector',    fa: 'بردار',    c: 'science',   v: [0.00, 0.18,  0.00, 0.04, 0.00, 0.00, 0.12, 0.96] },
    { id: 'matrix',    fa: 'ماتریس',   c: 'science',   v: [0.00, 0.22,  0.00, 0.00, 0.00, 0.00, 0.10, 0.95] },
    { id: 'neuron',    fa: 'نورون',    c: 'science',   v: [0.34, 0.12,  0.00, 0.00, 0.00, 0.16, 0.04, 0.90] },
    { id: 'equation',  fa: 'معادله',   c: 'science',   v: [0.00, 0.26, -0.06, 0.00, 0.00, 0.00, 0.04, 0.94] },
    { id: 'algorithm', fa: 'الگوریتم', c: 'science',   v: [0.00, 0.30,  0.08, 0.00, 0.00, 0.00, 0.16, 0.92] },
    { id: 'light',     fa: 'نور',      c: 'science',   v: [0.00, 0.22,  0.00, 0.08, 0.12, 0.18, 0.08, 0.86] }
  ];

  var WORDS = RAW_WORDS.map(function (w, idx) {
    return {
      idx: idx,
      id: w.id,
      fa: w.fa,
      cluster: w.c,
      colorHex: CLUSTERS[w.c].hex,
      colorInt: CLUSTERS[w.c].color,
      v: vNorm(w.v)
    };
  });

  var WORD_BY_ID = {};
  WORDS.forEach(function (w) { WORD_BY_ID[w.id] = w; });

  /* Static Layer-0 vector of polysemous "شیر": equal superposition of its 3 worlds */
  var PROBE_STATIC = vNorm([0.16, 0.06, 0.04, 0.02, 0.56, 0.58, 0.56, 0.04]);

  /* Build exact 3x8 orthonormal projection basis P0 and orthogonal complement P1 in R^8 */
  function orthonormalize(rows) {
    var out = [];
    for (var i = 0; i < rows.length; i++) {
      var u = rows[i].slice();
      for (var j = 0; j < out.length; j++) {
        var proj = vDot(u, out[j]);
        for (var k = 0; k < u.length; k++) u[k] -= proj * out[j][k];
      }
      out.push(vNorm(u));
    }
    return out;
  }

  /* Six mutually orthonormal 8D vectors: first 3 form primary 3D view P0,
     next 3 form orthogonal 8D slice P1 so P(phi) = cos(phi)*P0 + sin(phi)*P1
     stays strictly orthonormal (P P^T = I_3) at every slice angle! */
  var BASIS_6 = orthonormalize([
    [ 0.56,  0.08,  0.14, -0.78, -0.72,  0.76, -0.02, -0.06], /* X0 */
    [ 0.44,  0.88, -0.04,  0.22, -0.60, -0.58, -0.66,  0.64], /* Y0 */
    [ 0.10, -0.06,  0.86,  0.14,  0.50,  0.48, -0.82, -0.76], /* Z0 */
    [-0.42,  0.54,  0.28,  0.44, -0.30,  0.42,  0.54, -0.44], /* X1 */
    [ 0.62, -0.34,  0.46,  0.42,  0.34, -0.26,  0.34,  0.56], /* Y1 */
    [ 0.24,  0.28, -0.52, -0.24,  0.44,  0.52, -0.32,  0.42]  /* Z1 */
  ]);

  var RADIUS = 192;
  function project8to3(vec8, phiRad) {
    var cp = Math.cos(phiRad || 0), sp = Math.sin(phiRad || 0);
    var coords = [0, 0, 0];
    for (var axis = 0; axis < 3; axis++) {
      var b0 = BASIS_6[axis], b1 = BASIS_6[axis + 3];
      var val = 0;
      for (var d = 0; d < 8; d++) val += vec8[d] * (cp * b0[d] + sp * b1[d]);
      coords[axis] = val * RADIUS;
    }
    return coords;
  }

  /* =====================================================================
     2. EXPERIMENT DATASETS (ANALOGIES, ATTENTION SENTENCES, SOFTMAX PROMPTS)
     ===================================================================== */
  var ANALOGIES = [
    { id: 'king',    a: 'king',   b: 'man',   c: 'woman',  expected: 'queen',
      label: 'پادشاه − مرد + زن',      desc: 'بردار «جنسیت» بدون اینکه کسی تعریفش کند، به‌صورت خطی از داده‌ها بیرون می‌آید.' },
    { id: 'paris',   a: 'tehran', b: 'iran',  c: 'france', expected: 'paris',
      label: 'تهران − ایران + فرانسه', desc: 'جهتِ «پایتخت‌بودن» بین کشورها موازی است؛ همان بردار، فرانسه را به پاریس می‌برد.' },
    { id: 'tokyo',   a: 'tehran', b: 'iran',  c: 'japan',  expected: 'tokyo',
      label: 'تهران − ایران + ژاپن',   desc: 'همان جهت پایتخت‌بودن روی نقطه‌ی ژاپن می‌افتد و کنار توکیو فرود می‌آید.' },
    { id: 'prince',  a: 'prince', b: 'man',   c: 'woman',  expected: 'princess',
      label: 'شاهزاده − مرد + زن',     desc: 'بردار جنسیت فقط برای پادشاه نیست؛ روی شاهزاده هم همان انتقال موازی را می‌سازد.' }
  ];

  var SENTENCES = [
    {
      id: 'wildlife',
      title: 'حیات‌وحش',
      html: 'در سکوت <mark>جنگل</mark>، <span class="gmx-probe-hl">شیر</span> ناگهان <mark>غرش</mark> کرد و به سمت <mark>آهو</mark> پرید.',
      tokens: ['jungle', 'roar', 'deer', 'hunt'],
      weightsBias: [0.9, 1.15, 0.85, 0.55],
      color: 0xFF6A5A,
      colorHex: '#FF6A5A',
      targetAnchor: 'tiger'
    },
    {
      id: 'food',
      title: 'نوشیدنی',
      html: 'برای <mark>صبحانه</mark> یک <mark>فنجان</mark> <span class="gmx-probe-hl">شیر</span> <mark>داغ</mark> با <mark>عسل</mark> و <mark>نان</mark> نوشیدم.',
      tokens: ['breakfast', 'cup', 'warm', 'honey', 'bread'],
      weightsBias: [1.05, 0.95, 0.85, 0.9, 0.7],
      color: 0x3EE0A8,
      colorHex: '#3EE0A8',
      targetAnchor: 'coffee'
    },
    {
      id: 'water',
      title: 'لوله‌کشی',
      html: 'با <mark>آچار</mark> پیچ <span class="gmx-probe-hl">شیر</span> را سفت کرد تا <mark>چکه</mark> و <mark>قطره</mark> از <mark>لوله</mark> قطع شود.',
      tokens: ['wrench', 'drip', 'drop', 'pipe'],
      weightsBias: [1.0, 1.05, 0.95, 1.15],
      color: 0x5294FF,
      colorHex: '#5294FF',
      targetAnchor: 'pipe'
    },
    {
      id: 'none',
      title: 'بدون بافت (مبهم)',
      html: 'فقط واژه‌ی تنها: <span class="gmx-probe-hl">شیر</span> — هیچ کلمه‌ی دیگری در جمله نیست.',
      tokens: ['tiger', 'coffee', 'pipe'],
      weightsBias: [0.0, 0.0, 0.0],
      color: 0xE31B23,
      colorHex: '#E31B23',
      targetAnchor: null
    }
  ];

  var TEMP_PROMPTS = [
    {
      id: 'factual',
      title: 'پرامپت واقعی (دانشی)',
      prefix: 'پایتخت کشور فرانسه شهرِ',
      candidates: [
        { word: 'پاریس',   z: 9.2, kind: 'safe' },
        { word: 'لیون',    z: 5.0, kind: 'mid' },
        { word: 'مارسی',   z: 4.5, kind: 'mid' },
        { word: 'بوردو',   z: 3.9, kind: 'mid' },
        { word: 'لندن',    z: 3.4, kind: 'wild' },
        { word: 'تهران',   z: 2.8, kind: 'wild' },
        { word: 'توکیو',   z: 2.4, kind: 'wild' },
        { word: 'زیبای',   z: 2.0, kind: 'mid' },
        { word: 'قهوه',    z: 1.3, kind: 'wild' },
        { word: 'آچار',    z: 0.7, kind: 'wild' }
      ]
    },
    {
      id: 'creative',
      title: 'پرامپت خلاقانه (ادبی)',
      prefix: 'در سکوت شب، ستاره‌ها در آسمانِ',
      candidates: [
        { word: 'تاریک',    z: 6.2, kind: 'safe' },
        { word: 'بی‌کران',  z: 5.8, kind: 'safe' },
        { word: 'کویر',     z: 5.4, kind: 'safe' },
        { word: 'آبی',      z: 5.0, kind: 'safe' },
        { word: 'شهر',      z: 4.5, kind: 'mid' },
        { word: 'خیال',     z: 4.1, kind: 'mid' },
        { word: 'خاموش',    z: 3.7, kind: 'mid' },
        { word: 'شیشه‌ای',  z: 2.9, kind: 'mid' },
        { word: 'ماتریس',   z: 1.8, kind: 'wild' },
        { word: 'لوله‌کشی', z: 0.9, kind: 'wild' }
      ]
    }
  ];

  /* Real Scaled Dot-Product Attention for "شیر" at Transformer Layer L in [0, 12] */
  function computeAttentionState(sentId, layer) {
    var s = SENTENCES.find(function (x) { return x.id === sentId; }) || SENTENCES[0];
    var scores = [];
    var invSqrtD = 1 / Math.sqrt(8);
    var maxS = -Infinity;
    for (var i = 0; i < s.tokens.length; i++) {
      var w = WORD_BY_ID[s.tokens[i]];
      var sc = vDot(PROBE_STATIC, w.v) * invSqrtD + (s.weightsBias[i] || 0);
      scores.push(sc);
      if (sc > maxS) maxS = sc;
    }
    var exps = [], sumE = 0;
    for (var j = 0; j < scores.length; j++) {
      var e = Math.exp((scores[j] - maxS) * 2.1);
      exps.push(e); sumE += e;
    }
    var weights = exps.map(function (e) { return e / (sumE || 1); });

    /* Weighted value aggregation */
    var ctxVec = [0, 0, 0, 0, 0, 0, 0, 0];
    for (var k = 0; k < s.tokens.length; k++) {
      var wv = WORD_BY_ID[s.tokens[k]].v;
      for (var d = 0; d < 8; d++) ctxVec[d] += weights[k] * wv[d];
    }
    ctxVec = vNorm(ctxVec);

    /* Residual stream update across layers 0..12:
       At layer 0 => static ambiguous vector; by layer 12 => deeply contextualized */
    var gamma = s.id === 'none' ? 0 : (1 - Math.exp(-0.29 * layer)) / (1 - Math.exp(-0.29 * 12));
    var outVec = vNorm(vLerp(PROBE_STATIC, ctxVec, gamma * 0.94));

    return {
      sentence: s,
      weights: weights,
      gamma: gamma,
      vec: outVec,
      affinities: {
        wildlife: Math.max(0, cosSim(outVec, WORD_BY_ID.tiger.v)),
        food:     Math.max(0, cosSim(outVec, WORD_BY_ID.coffee.v)),
        water:    Math.max(0, cosSim(outVec, WORD_BY_ID.pipe.v))
      }
    };
  }

  /* Real Softmax with Temperature T & Top-P Nucleus filtering */
  function computeSoftmax(promptId, T, topP) {
    var pr = TEMP_PROMPTS.find(function (p) { return p.id === promptId; }) || TEMP_PROMPTS[0];
    var temp = Math.max(0.05, T);
    var maxZ = -Infinity;
    pr.candidates.forEach(function (c) { if (c.z > maxZ) maxZ = c.z; });

    var raw = pr.candidates.map(function (c, i) {
      return { idx: i, word: c.word, z: c.z, kind: c.kind, e: Math.exp((c.z - maxZ) / temp) };
    });
    var sum = raw.reduce(function (acc, r) { return acc + r.e; }, 0) || 1;
    raw.forEach(function (r) { r.p = r.e / sum; });

    /* Sort copy descending for exact Top-P nucleus cutoff */
    var sorted = raw.slice().sort(function (a, b) { return b.p - a.p; });
    var cum = 0;
    var activeSet = {};
    for (var i = 0; i < sorted.length; i++) {
      activeSet[sorted[i].idx] = true;
      cum += sorted[i].p;
      if (cum >= topP && i >= 0) break;
    }

    var activeSum = 0;
    raw.forEach(function (r) {
      r.inNucleus = !!activeSet[r.idx];
      if (r.inNucleus) activeSum += r.p;
    });
    raw.forEach(function (r) {
      r.pEff = r.inNucleus ? (r.p / (activeSum || 1)) : 0;
    });

    /* Shannon Entropy H (in bits) of the active sampling distribution */
    var H = 0;
    raw.forEach(function (r) {
      if (r.pEff > 1e-9) H -= r.pEff * (Math.log(r.pEff) / Math.LN2);
    });
    var wildRisk = raw.reduce(function (acc, r) { return acc + (r.kind === 'wild' ? r.pEff : 0); }, 0);

    return {
      prompt: pr,
      items: raw,
      entropy: H,
      perplexity: Math.pow(2, H),
      wildRisk: wildRisk
    };
  }

  /* =====================================================================
     3. CHAPTER DEFINITIONS
     ===================================================================== */
  var CHAPTERS = [
    {
      id: 'space', no: '۰۱', title: 'نقشه‌ی معنا',
      camera: { target: [0, 0, 0], dist: 455, phi: 1.24, theta: 0.42 },
      story: 'در ذهن یک مدل زبانی، هیچ واژه‌ای «کلمه» نیست — یک <b>نقطه با مختصات عددی</b> روی کُره‌ای چندصدبعدی است. اینجا ۳۶ واژه‌ی فارسی را می‌بینی که با بردار ۸بعدی واقعی روی فضا پخش شده‌اند: کلمه‌های هم‌خانواده بی‌آنکه کسی دسته‌بندی‌شان کند، کنار هم خوشه می‌سازند. <b>روی هر کلمه در صحنه‌ی سه‌بعدی کلیک کن</b> تا بردار ۸بعدی و زاویه‌ی کسینوسی همسایه‌هایش را ببینی.'
    },
    {
      id: 'algebra', no: '۰۲', title: 'جبرِ معنا',
      camera: { target: [18, 72, 6], dist: 365, phi: 1.18, theta: 0.72 },
      story: 'اگر معنا «موقعیت در فضا» باشد، می‌توان با کلمات <b>جمع و تفریق هندسی</b> کرد! در صحنه‌ی سه‌بعدی، پیکان قرمز جهت تفاوت از <b>«مرد» به «زن»</b> را اندازه می‌گیرد و پیکان آبی دقیقاً همان بردار را به‌شکل موازی روی <b>«پادشاه»</b> می‌گذارد — و نقطه دقیقاً در آغوش <b>«ملکه»</b> فرود می‌آید. معادله‌ی پایتخت‌ها را هم امتحان کن و اسلایدر انتقال برداری را بکش.'
    },
    {
      id: 'attention', no: '۰۳', title: 'توجه و معمای «شیر»',
      camera: { target: [0, -74, 14], dist: 335, phi: 1.16, theta: 0.22 },
      story: 'واژه‌ی <b>«شیر»</b> به تنهایی مبهم است؛ پس در لایه‌ی صفر وسط مثلثِ سه جهان (حیوان، نوشیدنی، لوله‌کشی) معلق می‌ماند. حالا یکی از جمله‌های پایین را انتخاب کن و اسلایدر <b>لایه‌ی ترنسفورمر (۰ تا ۱۲)</b> را جلو ببر: پرتوهای <b>توجه (Attention)</b> از کلمه‌های بافت جمله به سمت «شیر» شلیک می‌شوند و مختصات آن را واقعاً در فضای سه‌بعدی به خوشه‌ی درست می‌کشند.'
    },
    {
      id: 'temperature', no: '۰۴', title: 'دما و ریشه‌ی توهم',
      camera: { target: [0, 32, 0], dist: 390, phi: 1.12, theta: 0.32 },
      story: 'هوش مصنوعی «حقیقت» را نمی‌داند؛ برای کلمه‌ی بعدی یک <b>توزیع احتمال (Softmax)</b> می‌سازد و از آن نمونه‌گیری می‌کند. اسلایدر <b>دما (Temperature)</b> را بالا ببر و چند بار دکمه‌ی <b>«تولید کلمه‌ی بعدی»</b> را بزن: در دمای پایین همیشه امن‌ترین ستون برنده است، اما با بالا رفتن دما، ستون‌های قرمزِ دُمِ توزیع (مثل «تهران» یا «آچار» برای پایتخت فرانسه!) جان می‌گیرند — این ریشه‌ی ریاضی <b>توهم (Hallucination)</b> است.'
    },
    {
      id: 'limits', no: '۰۵', title: 'سایه‌ی ۱۲٬۰۰۰ بُعد',
      camera: { target: [0, 0, 0], dist: 450, phi: 1.28, theta: 1.05 },
      story: 'یک نکته‌ی صادقانه: مدل‌های واقعی ۲۰۲۶ در فضایی با <b>چندهزار بُعد</b> زندگی می‌کنند، اما چشم ما فقط ۳ بُعد را می‌بیند. اسلایدر <b>«زاویه‌ی برش ابعاد پنهان»</b> را بچرخان: هیچ عددی در بردارهای ۸بعدی عوض نمی‌شود، اما می‌بینی کلمه‌هایی که در یک زاویه به هم نزدیک به‌نظر می‌رسیدند، در زاویه‌ی دیگر کاملاً از هم جدا می‌شوند. آنچه روی صفحه می‌بینی همیشه «سایه‌ی» یک هندسه‌ی بسیار بزرگ‌تر است.'
    }
  ];

  /* =====================================================================
     4. PROCEDURAL GLOW TEXTURE HELPER (ZERO EXTRA NETWORK FILES)
     ===================================================================== */
  function makeGlowTexture(THREE, innerRgba, midRgba) {
    var sz = 128;
    var c = document.createElement('canvas');
    c.width = sz; c.height = sz;
    var g = c.getContext('2d');
    var grd = g.createRadialGradient(sz / 2, sz / 2, 0, sz / 2, sz / 2, sz / 2);
    grd.addColorStop(0.0, innerRgba || 'rgba(255,255,255,1)');
    grd.addColorStop(0.28, midRgba || 'rgba(255,255,255,0.45)');
    grd.addColorStop(0.62, 'rgba(255,255,255,0.09)');
    grd.addColorStop(1.0, 'rgba(255,255,255,0)');
    g.fillStyle = grd;
    g.fillRect(0, 0, sz, sz);
    var tex = new THREE.CanvasTexture(c);
    tex.needsUpdate = true;
    return tex;
  }

  /* =====================================================================
     5. MAIN COCKPIT & THREE.JS ENGINE BOOTSTRAP
     ===================================================================== */
  function mountGeometryLab(rootSel) {
    var root = document.querySelector(rootSel);
    if (!root) return;

    var fsWrap     = root.querySelector('.gmx-fs');
    var stage      = root.querySelector('.gmx-stage');
    var canvas     = root.querySelector('.gmx-canvas');
    var noteOverlay= root.querySelector('.xstage__note');
    var labelLayer = root.querySelector('.gmx-labels');
    var hintEl     = root.querySelector('.gmx-hint');

    var chButtons  = Array.prototype.slice.call(root.querySelectorAll('[data-gm-ch]'));
    var storyEl    = root.querySelector('[data-gm-story]');
    var telemetryEl= root.querySelector('[data-gm-telemetry]');
    var inspectorEl= root.querySelector('[data-gm-inspector]');

    var panelAlgebra   = root.querySelector('[data-gm-panel="algebra"]');
    var panelAttention = root.querySelector('[data-gm-panel="attention"]');
    var panelTemp      = root.querySelector('[data-gm-panel="temperature"]');
    var panelLimits    = root.querySelector('[data-gm-panel="limits"]');
    var panelFreeTabs  = root.querySelector('[data-gm-panel="freetabs"]');

    /* Application state */
    var state = {
      chapter: 'space',
      freeMode: false,
      freeSub: 'attention',
      running: false,
      ready: false,
      dead: false,
      autoOrbit: true,
      dockHidden: false,
      selectedWordId: null,
      hoverWordId: null,

      /* Analogy state */
      analogyId: 'king',
      analogyProgress: 1.0,

      /* Attention state */
      sentenceId: 'wildlife',
      layer: 9.0,

      /* Temperature state */
      promptId: 'factual',
      temperature: 0.85,
      topP: 0.92,
      sampledIdx: 0,
      sampleHistory: [],
      sampleFlash: 0,

      /* 8D Slice state */
      sliceDeg: 0
    };

    /* Read initial ?c= chapter query param if present */
    try {
      var qCh = new URL(location.href).searchParams.get('c');
      if (qCh && (qCh === 'free' || CHAPTERS.some(function (x) { return x.id === qCh; }))) {
        state.chapter = qCh;
        state.freeMode = (qCh === 'free');
      }
    } catch (_) {}

    /* =====================================================================
       UI BINDINGS (FULLSCREEN, CONTROLS, MODE PANELS, INSPECTOR)
       ===================================================================== */
    var fauxFull = false;
    var fsPlaceholder = document.createComment('gmx-fs-placeholder');
    var canNativeFS = !!(document.fullscreenEnabled || document.webkitFullscreenEnabled) &&
                      !!(fsWrap.requestFullscreen || fsWrap.webkitRequestFullscreen);

    function isFull() {
      return fauxFull || (document.fullscreenElement || document.webkitFullscreenElement) === fsWrap;
    }

    var fsButtons = Array.prototype.slice.call(root.querySelectorAll('[data-gm-fs]'));
    var playButtons = Array.prototype.slice.call(root.querySelectorAll('[data-gm-play]'));
    var orbitBtn = root.querySelector('[data-gm-orbit]');
    var dockBtn = root.querySelector('[data-gm-dock]');

    function portalToBody(on) {
      if (on) {
        if (fsWrap.parentNode && fsWrap.parentNode !== document.body) {
          fsWrap.parentNode.insertBefore(fsPlaceholder, fsWrap);
          fsWrap.setAttribute('data-nvx-nolift', '1');
          document.body.appendChild(fsWrap);
        }
      } else {
        if (fsPlaceholder.parentNode) {
          fsPlaceholder.parentNode.insertBefore(fsWrap, fsPlaceholder);
          fsPlaceholder.parentNode.removeChild(fsPlaceholder);
          fsWrap.removeAttribute('data-nvx-nolift');
        }
      }
    }

    function paintFullButtons() {
      var on = isFull();
      portalToBody(fauxFull);
      fsWrap.classList.toggle('is-full', on);
      fsWrap.classList.toggle('is-faux', fauxFull);
      document.documentElement.classList.toggle('gmx-locked', on);
      fsButtons.forEach(function (b) {
        b.setAttribute('aria-pressed', String(on));
        var sp = b.querySelector('span');
        if (sp) sp.textContent = on ? 'خروج از تمام‌صفحه' : 'تمام‌صفحه';
      });
      onResize();
      setTimeout(onResize, 60);
    }

    function toggleFullscreen(forceFaux) {
      if (fauxFull) { fauxFull = false; paintFullButtons(); return; }
      if (document.fullscreenElement || document.webkitFullscreenElement) {
        (document.exitFullscreen || document.webkitExitFullscreen).call(document);
        return;
      }
      if (forceFaux || !canNativeFS) { fauxFull = true; paintFullButtons(); return; }
      try {
        var req = (fsWrap.requestFullscreen || fsWrap.webkitRequestFullscreen).call(fsWrap);
        if (req && req.catch) req.catch(function () { fauxFull = true; paintFullButtons(); });
      } catch (_) {
        fauxFull = true; paintFullButtons();
      }
    }

    fsButtons.forEach(function (b) { b.addEventListener('click', function () { toggleFullscreen(false); }); });
    ['fullscreenchange', 'webkitfullscreenchange'].forEach(function (ev) {
      document.addEventListener(ev, function () { if (!fauxFull) paintFullButtons(); });
    });

    if (dockBtn) {
      dockBtn.addEventListener('click', function () {
        state.dockHidden = !state.dockHidden;
        fsWrap.classList.toggle('is-dock-hidden', state.dockHidden);
        dockBtn.setAttribute('aria-pressed', String(state.dockHidden));
        var sp = dockBtn.querySelector('span');
        if (sp) sp.textContent = state.dockHidden ? 'نمایش پنل' : 'مخفی‌کردن پنل';
      });
    }

    if (orbitBtn) {
      orbitBtn.setAttribute('aria-pressed', String(state.autoOrbit));
      orbitBtn.addEventListener('click', function () {
        state.autoOrbit = !state.autoOrbit;
        orbitBtn.setAttribute('aria-pressed', String(state.autoOrbit));
      });
    }

    function paintPlayButtons() {
      playButtons.forEach(function (b) {
        b.setAttribute('aria-pressed', String(state.running));
        var sp = b.querySelector('span');
        if (sp) sp.textContent = state.running ? 'توقف' : 'اجرا';
      });
    }
    playButtons.forEach(function (b) {
      b.addEventListener('click', function () { state.running ? pause() : play(); });
    });

    Array.prototype.slice.call(root.querySelectorAll('[data-gm-reset]')).forEach(function (b) {
      b.addEventListener('click', function () {
        state.selectedWordId = null;
        state.analogyId = 'king';
        state.analogyProgress = 1.0;
        state.sentenceId = 'wildlife';
        state.layer = 9.0;
        state.promptId = 'factual';
        state.temperature = 0.85;
        state.topP = 0.92;
        state.sliceDeg = 0;
        state.sampleHistory = [];
        state.autoOrbit = true;
        if (orbitBtn) orbitBtn.setAttribute('aria-pressed', 'true');
        syncAllSliders();
        closeInspector();
        setChapter('space', false);
      });
    });

    /* Inspector card rendering */
    function closeInspector() {
      state.selectedWordId = null;
      if (inspectorEl) inspectorEl.hidden = true;
      updateTelemetry();
    }

    function openInspector(wordId) {
      var w = WORD_BY_ID[wordId];
      if (!w || !inspectorEl) return;
      state.selectedWordId = wordId;

      /* Compute real 8D cosine similarity against all other 35 words */
      var ranked = WORDS.filter(function (o) { return o.id !== w.id; }).map(function (o) {
        var sim = cosSim(w.v, o.v);
        return { word: o, sim: sim, deg: Math.acos(sim) * 180 / Math.PI };
      }).sort(function (a, b) { return b.sim - a.sim; });

      var top3 = ranked.slice(0, 3);
      var farthest = ranked[ranked.length - 1];

      var barsHtml = w.v.map(function (val, i) {
        var mag = Math.min(46, Math.abs(val) * 48);
        var topPct = val >= 0 ? (50 - mag) : 50;
        var bg = val >= 0 ? w.colorHex : '#E31B23';
        return '<div class="gmx-vec-col" title="' + DIM_NAMES[i] + ': ' + val.toFixed(2) + '">' +
               '<div class="gmx-vec-fill" style="top:' + topPct + '%;height:' + Math.max(3, mag) + '%;background:' + bg + '"></div>' +
               '</div>';
      }).join('');

      var nbHtml = top3.map(function (item) {
        return '<button type="button" class="gmx-nb-btn" data-jump-word="' + item.word.id + '">' +
               '<span>' + item.word.fa + '</span>' +
               '<span class="gmx-nb-meta">cos ' + fa(item.sim.toFixed(2)) + ' · ' + fa(Math.round(item.deg)) + '°</span>' +
               '</button>';
      }).join('');

      inspectorEl.innerHTML =
        '<div class="gmx-insp__head">' +
          '<div class="gmx-insp__title">' +
            '<span class="gmx-insp__dot" style="background:' + w.colorHex + ';box-shadow:0 0 12px ' + w.colorHex + '"></span>' +
            '<span class="gmx-insp__word">' + w.fa + '</span>' +
            '<span class="gmx-insp__cluster">' + CLUSTERS[w.cluster].name + '</span>' +
          '</div>' +
          '<button type="button" class="gmx-insp__close" aria-label="بستن" data-insp-close>×</button>' +
        '</div>' +
        '<p class="gmx-insp__sec-title">اثرانگشت بردار ۸بعدی (ابعاد نهفته):</p>' +
        '<div class="gmx-vec-bars">' + barsHtml + '</div>' +
        '<p class="gmx-insp__sec-title">نزدیک‌ترین همسایه‌ها در ۸ بُعد (کلیک برای پرش):</p>' +
        '<div class="gmx-insp__neighbors">' + nbHtml + '</div>' +
        '<p class="gmx-insp__sec-title" style="margin-top:8px;margin-bottom:0">دورترین واژه: <b style="color:#fff">' + farthest.word.fa + '</b> (زاویه ' + fa(Math.round(farthest.deg)) + '°)</p>';

      inspectorEl.hidden = false;

      var closeBtn = inspectorEl.querySelector('[data-insp-close]');
      if (closeBtn) closeBtn.addEventListener('click', closeInspector);
      Array.prototype.slice.call(inspectorEl.querySelectorAll('[data-jump-word]')).forEach(function (btn) {
        btn.addEventListener('click', function () {
          var id = btn.getAttribute('data-jump-word');
          openInspector(id);
          focusWordInCamera(id);
        });
      });
      updateTelemetry();
    }

    function activeMode() {
      return state.freeMode ? state.freeSub : state.chapter;
    }

    /* ---------- Mode Panel Builders & Updaters ---------- */
    var eqFormulaEl = root.querySelector('[data-gm-eq-formula]');
    var eqSimEl     = root.querySelector('[data-gm-eq-sim]');
    var analogyPills = Array.prototype.slice.call(root.querySelectorAll('[data-gm-analogy]'));

    function paintAnalogyPanel() {
      var an = ANALOGIES.find(function (x) { return x.id === state.analogyId; }) || ANALOGIES[0];
      var wa = WORD_BY_ID[an.a], wb = WORD_BY_ID[an.b], wc = WORD_BY_ID[an.c], wt = WORD_BY_ID[an.expected];
      var resVec = vNorm(vAdd(vSub(wa.v, wb.v), wc.v));
      var sim = cosSim(resVec, wt.v);
      var deg = angleDeg(resVec, wt.v);

      analogyPills.forEach(function (b) {
        b.setAttribute('aria-pressed', String(b.getAttribute('data-gm-analogy') === an.id));
      });
      if (eqFormulaEl) {
        eqFormulaEl.innerHTML =
          '<span>' + wa.fa + '</span><span class="gmx-eq-op">−</span>' +
          '<span>' + wb.fa + '</span><span class="gmx-eq-op">+</span>' +
          '<span>' + wc.fa + '</span><span class="gmx-eq-op">≈</span>' +
          '<span class="gmx-eq-res">' + wt.fa + '</span>';
      }
      if (eqSimEl) {
        eqSimEl.innerHTML = 'شباهت کسینوسی واقعی: <b>' + fa(sim.toFixed(3)) + '</b> · زاویه ' + fa(deg.toFixed(1)) + '°';
      }
      updateTelemetry();
    }

    analogyPills.forEach(function (b) {
      b.addEventListener('click', function () {
        state.analogyId = b.getAttribute('data-gm-analogy');
        state.analogyProgress = 1.0;
        syncAllSliders();
        paintAnalogyPanel();
        focusAnalogyCamera();
      });
    });

    /* Attention panel */
    var sentPills = Array.prototype.slice.call(root.querySelectorAll('[data-gm-sent]'));
    var affEls = {
      wildlifeVal: root.querySelector('[data-gm-aff-val="wildlife"]'),
      wildlifeBar: root.querySelector('[data-gm-aff-bar="wildlife"]'),
      foodVal:     root.querySelector('[data-gm-aff-val="food"]'),
      foodBar:     root.querySelector('[data-gm-aff-bar="food"]'),
      waterVal:    root.querySelector('[data-gm-aff-val="water"]'),
      waterBar:    root.querySelector('[data-gm-aff-bar="water"]')
    };

    function paintAttentionPanel() {
      var att = computeAttentionState(state.sentenceId, state.layer);
      sentPills.forEach(function (b) {
        b.setAttribute('aria-pressed', String(b.getAttribute('data-gm-sent') === state.sentenceId));
      });
      ['wildlife', 'food', 'water'].forEach(function (k) {
        var pct = Math.round(att.affinities[k] * 100);
        if (affEls[k + 'Val']) affEls[k + 'Val'].textContent = fa(pct) + '٪';
        if (affEls[k + 'Bar']) affEls[k + 'Bar'].style.width = pct + '%';
      });
      updateTelemetry();
    }

    sentPills.forEach(function (b) {
      b.addEventListener('click', function () {
        state.sentenceId = b.getAttribute('data-gm-sent');
        if (state.sentenceId === 'none') {
          state.layer = 0;
        } else if (state.layer < 2) {
          state.layer = 9;
        }
        syncAllSliders();
        paintAttentionPanel();
        resetProbeTrail();
      });
    });

    /* Temperature & Softmax panel */
    var promptPills   = Array.prototype.slice.call(root.querySelectorAll('[data-gm-prompt]'));
    var probListEl    = root.querySelector('[data-gm-probs]');
    var promptPrevEl  = root.querySelector('[data-gm-prompt-preview]');
    var sampleHistEl  = root.querySelector('[data-gm-sample-hist]');
    var entropyValEl  = root.querySelector('[data-gm-entropy]');
    var riskValEl     = root.querySelector('[data-gm-risk]');
    var sampleBtn     = root.querySelector('[data-gm-sample-btn]');

    function paintTempPanel() {
      var sm = computeSoftmax(state.promptId, state.temperature, state.topP);
      promptPills.forEach(function (b) {
        b.setAttribute('aria-pressed', String(b.getAttribute('data-gm-prompt') === state.promptId));
      });
      if (probListEl) {
        probListEl.innerHTML = sm.items.map(function (it) {
          var pct = (it.pEff * 100);
          var rawPct = (it.p * 100);
          var barColor = it.kind === 'safe' ? '#3EE0A8' : (it.kind === 'mid' ? '#F5B754' : '#E31B23');
          var cls = 'gmx-prob-row' +
                    (it.inNucleus ? '' : ' is-cut') +
                    (it.kind === 'wild' ? ' is-hallucination' : '') +
                    (it.idx === state.sampledIdx ? ' is-sampled' : '');
          return '<div class="' + cls + '">' +
                 '<span class="gmx-prob-word">' + it.word + '</span>' +
                 '<span class="gmx-prob-track"><span class="gmx-prob-fill" style="width:' + Math.max(0.6, rawPct).toFixed(1) + '%;background:' + barColor + '"></span></span>' +
                 '<span class="gmx-prob-pct">' + (it.inNucleus ? fa(pct.toFixed(1)) + '٪' : 'حذف') + '</span>' +
                 '</div>';
        }).join('');
      }
      var chosen = sm.items[state.sampledIdx] || sm.items[0];
      if (promptPrevEl) {
        var tokCls = 'gmx-sampled-token' + (chosen.kind === 'safe' ? ' is-safe' : (chosen.kind === 'wild' ? ' is-wild' : ''));
        promptPrevEl.innerHTML = sm.prompt.prefix + ' <span class="' + tokCls + '">' + chosen.word + '</span>';
      }
      if (entropyValEl) entropyValEl.textContent = fa(sm.entropy.toFixed(2)) + ' بیت';
      if (riskValEl) {
        var riskPct = Math.round(sm.wildRisk * 100);
        riskValEl.textContent = fa(riskPct) + '٪';
        riskValEl.style.color = riskPct >= 25 ? '#FF6A5A' : (riskPct >= 8 ? '#F5B754' : '#3EE0A8');
      }
      if (sampleHistEl) {
        sampleHistEl.innerHTML = state.sampleHistory.slice(-7).map(function (h) {
          return '<span class="gmx-hist-chip' + (h.kind === 'wild' ? ' is-wild' : '') + '">' + h.word + '</span>';
        }).join('');
      }
      updateTelemetry();
    }

    function runRealTokenSample() {
      var sm = computeSoftmax(state.promptId, state.temperature, state.topP);
      var r = Math.random(), acc = 0, pick = sm.items[0];
      for (var i = 0; i < sm.items.length; i++) {
        if (!sm.items[i].inNucleus) continue;
        acc += sm.items[i].pEff;
        if (r <= acc) { pick = sm.items[i]; break; }
      }
      state.sampledIdx = pick.idx;
      state.sampleFlash = 1.0;
      state.sampleHistory.push({ word: pick.word, kind: pick.kind });
      if (state.sampleHistory.length > 14) state.sampleHistory.shift();
      paintTempPanel();
    }

    promptPills.forEach(function (b) {
      b.addEventListener('click', function () {
        state.promptId = b.getAttribute('data-gm-prompt');
        state.sampledIdx = 0;
        state.sampleHistory = [];
        paintTempPanel();
      });
    });
    if (sampleBtn) sampleBtn.addEventListener('click', runRealTokenSample);

    /* Free Lab sub-mode tabs */
    var freeTabBtns = Array.prototype.slice.call(root.querySelectorAll('[data-gm-freesub]'));
    freeTabBtns.forEach(function (b) {
      b.addEventListener('click', function () {
        state.freeSub = b.getAttribute('data-gm-freesub');
        updateVisiblePanels();
      });
    });

    /* Sliders */
    var slAnalogy = root.querySelector('[data-gm-sl="analogy"]');
    var slLayer   = root.querySelector('[data-gm-sl="layer"]');
    var slTemp    = root.querySelector('[data-gm-sl="temp"]');
    var slTopP    = root.querySelector('[data-gm-sl="topp"]');
    var slSlice   = root.querySelector('[data-gm-sl="slice"]');

    function setSlVal(key, text) {
      var el = root.querySelector('[data-gm-val="' + key + '"]');
      if (el) el.textContent = text;
    }

    function syncAllSliders() {
      if (slAnalogy) { slAnalogy.value = state.analogyProgress; setSlVal('analogy', fa(Math.round(state.analogyProgress * 100)) + '٪'); }
      if (slLayer)   { slLayer.value = state.layer; setSlVal('layer', 'لایه ' + fa(Math.round(state.layer)) + ' از ۱۲'); }
      if (slTemp)    { slTemp.value = state.temperature; setSlVal('temp', 'T = ' + fa(state.temperature.toFixed(2))); }
      if (slTopP)    { slTopP.value = state.topP; setSlVal('topp', 'Top-P = ' + fa(Math.round(state.topP * 100)) + '٪'); }
      if (slSlice)   { slSlice.value = state.sliceDeg; setSlVal('slice', fa(Math.round(state.sliceDeg)) + '°'); }
    }

    if (slAnalogy) slAnalogy.addEventListener('input', function () {
      state.analogyProgress = parseFloat(slAnalogy.value);
      setSlVal('analogy', fa(Math.round(state.analogyProgress * 100)) + '٪');
      updateTelemetry();
    });
    if (slLayer) slLayer.addEventListener('input', function () {
      state.layer = parseFloat(slLayer.value);
      setSlVal('layer', 'لایه ' + fa(Math.round(state.layer)) + ' از ۱۲');
      paintAttentionPanel();
    });
    if (slTemp) slTemp.addEventListener('input', function () {
      state.temperature = parseFloat(slTemp.value);
      setSlVal('temp', 'T = ' + fa(state.temperature.toFixed(2)));
      paintTempPanel();
    });
    if (slTopP) slTopP.addEventListener('input', function () {
      state.topP = parseFloat(slTopP.value);
      setSlVal('topp', 'Top-P = ' + fa(Math.round(state.topP * 100)) + '٪');
      paintTempPanel();
    });
    if (slSlice) slSlice.addEventListener('input', function () {
      state.sliceDeg = parseFloat(slSlice.value);
      setSlVal('slice', fa(Math.round(state.sliceDeg)) + '°');
      updateTelemetry();
    });

    function updateVisiblePanels() {
      var m = activeMode();
      if (panelFreeTabs) panelFreeTabs.hidden = !state.freeMode;
      freeTabBtns.forEach(function (b) {
        b.setAttribute('aria-pressed', String(b.getAttribute('data-gm-freesub') === state.freeSub));
      });
      if (panelAlgebra)   panelAlgebra.hidden   = (m !== 'algebra');
      if (panelAttention) panelAttention.hidden = (m !== 'attention');
      if (panelTemp)      panelTemp.hidden      = (m !== 'temperature');
      if (panelLimits)    panelLimits.hidden    = (m !== 'limits');

      if (m === 'algebra') paintAnalogyPanel();
      if (m === 'attention') paintAttentionPanel();
      if (m === 'temperature') paintTempPanel();
      updateTelemetry();
    }

    function updateTelemetry() {
      if (!telemetryEl) return;
      var m = activeMode();
      var chips = [];
      if (m === 'space') {
        chips.push('<span class="gmx-chip gmx-chip--ion">بُعد واقعی: <strong>۸ بُعد ← ۳ بُعد</strong></span>');
        chips.push('<span class="gmx-chip">واژگان: <strong>۳۶ واژه · ۶ خوشه</strong></span>');
        if (state.selectedWordId && WORD_BY_ID[state.selectedWordId]) {
          chips.push('<span class="gmx-chip gmx-chip--accent">انتخاب‌شده: <strong>' + WORD_BY_ID[state.selectedWordId].fa + '</strong></span>');
        }
      } else if (m === 'algebra') {
        var an = ANALOGIES.find(function (x) { return x.id === state.analogyId; }) || ANALOGIES[0];
        var wa = WORD_BY_ID[an.a], wb = WORD_BY_ID[an.b], wc = WORD_BY_ID[an.c], wt = WORD_BY_ID[an.expected];
        var rv = vNorm(vAdd(vSub(wa.v, wb.v), wc.v));
        chips.push('<span class="gmx-chip gmx-chip--accent">مقصد: <strong>' + wt.fa + '</strong></span>');
        chips.push('<span class="gmx-chip gmx-chip--ion">cos(θ): <strong>' + fa(cosSim(rv, wt.v).toFixed(3)) + '</strong></span>');
      } else if (m === 'attention') {
        var att = computeAttentionState(state.sentenceId, state.layer);
        chips.push('<span class="gmx-chip gmx-chip--accent">بافت «شیر»: <strong>' + att.sentence.title + '</strong></span>');
        chips.push('<span class="gmx-chip gmx-chip--ion">لایه: <strong>' + fa(Math.round(state.layer)) + ' / ۱۲</strong></span>');
      } else if (m === 'temperature') {
        var sm = computeSoftmax(state.promptId, state.temperature, state.topP);
        chips.push('<span class="gmx-chip gmx-chip--accent">دما: <strong>T = ' + fa(state.temperature.toFixed(2)) + '</strong></span>');
        chips.push('<span class="gmx-chip gmx-chip--ion">آنتروپی: <strong>' + fa(sm.entropy.toFixed(2)) + ' بیت</strong></span>');
        chips.push('<span class="gmx-chip">سرگشتگی: <strong>' + fa(sm.perplexity.toFixed(1)) + '</strong></span>');
      } else if (m === 'limits') {
        var pKing = project8to3(WORD_BY_ID.king.v, state.sliceDeg * Math.PI / 180);
        var pPipe = project8to3(WORD_BY_ID.pipe.v, state.sliceDeg * Math.PI / 180);
        var d3 = Math.hypot(pKing[0] - pPipe[0], pKing[1] - pPipe[1], pKing[2] - pPipe[2]) / RADIUS;
        var deg8 = angleDeg(WORD_BY_ID.king.v, WORD_BY_ID.pipe.v);
        chips.push('<span class="gmx-chip gmx-chip--ion">برش SO(8): <strong>' + fa(Math.round(state.sliceDeg)) + '°</strong></span>');
        chips.push('<span class="gmx-chip gmx-chip--accent">زاویه واقعی ۸بُعدی: <strong>' + fa(Math.round(deg8)) + '° (ثابت)</strong></span>');
        chips.push('<span class="gmx-chip">فاصله‌ی ظاهری ۳بُعدی: <strong>' + fa(d3.toFixed(2)) + '</strong></span>');
      }
      telemetryEl.innerHTML = chips.join('');
    }

    function setChapter(id, pushUrl) {
      state.freeMode = (id === 'free');
      if (!state.freeMode) state.chapter = id;

      chButtons.forEach(function (b) {
        b.setAttribute('aria-current', String(b.getAttribute('data-gm-ch') === id));
      });

      if (state.freeMode) {
        if (storyEl) {
          storyEl.innerHTML = '<b>آزمایشگاه آزاد فعال است:</b> هر ابزاری را می‌خواهی پایین انتخاب کن، روی هر واژه در فضا کلیک کن، دوربین را آزادانه بچرخان و از دکمه‌ی <b>تمام‌صفحه</b> برای تجربه‌ی سینمایی استفاده کن.';
        }
      } else {
        var ch = CHAPTERS.find(function (x) { return x.id === id; }) || CHAPTERS[0];
        if (storyEl) storyEl.innerHTML = ch.story;
        if (orbit && ch.camera) orbit.goTo(ch.camera);
        if (id === 'limits') {
          state.sliceDeg = 0;
          syncAllSliders();
        }
        if (id === 'attention') resetProbeTrail();
      }

      if (id === 'temperature') closeInspector();
      updateVisiblePanels();

      if (pushUrl) {
        try {
          var u = new URL(location.href);
          u.searchParams.set('c', id);
          history.replaceState(null, '', u);
        } catch (_) {}
      }
    }

    chButtons.forEach(function (b) {
      b.addEventListener('click', function () {
        setChapter(b.getAttribute('data-gm-ch'), true);
      });
    });

    /* Keyboard shortcuts */
    document.addEventListener('keydown', function (e) {
      if (e.target && /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
      if (e.key === 'Escape') {
        if (fauxFull) { fauxFull = false; paintFullButtons(); }
        else if (state.selectedWordId) closeInspector();
      } else if (e.key === 'f' || e.key === 'F') {
        if (visible) toggleFullscreen();
      }
    });

    /* =====================================================================
       6. THREE.JS SCENE CONSTRUCTION & INTERACTIVE ENGINE
       ===================================================================== */
    var THREE = null, renderer = null, scene = null, camera = null;
    var orbit = null, raycaster = null;
    var raf = 0, lastTs = 0, clock = 0, visible = false, slowFrames = 0;

    /* 3D Object Registries */
    var galaxyGroup = null;
    var wordMeshes = [];
    var wordAuras = [];
    var wordHitTargets = [];
    var wordPositions = [];
    var filamentLines = null;
    var neighborLines = null;
    var sphereShell = null;
    var dustPoints = null;

    /* Chapter 02 Analogy Objects */
    var algebraGroup = null;
    var arrowSub = null, arrowAdd = null, dashedParallel = null;
    var ghostOrb = null, targetRing = null;

    /* Chapter 03 Attention Objects */
    var attentionGroup = null;
    var probeMesh = null, probeRing1 = null, probeRing2 = null, probeAura = null;
    var probeCurrentPos = null;
    var attCurves = [];
    var attLineMeshes = [];
    var attPulseMeshes = [];
    var probeTrailLine = null, probeTrailArr = null, probeTrailCount = 0;

    /* Chapter 04 Temperature Arena Objects */
    var tempGroup = null;
    var tempCore = null, tempCoreRing = null;
    var tempPillars = [];
    var tempTopPRing = null;
    var tempLaserLine = null, tempShockRing = null;
    var tempLabels = [];

    /* Chapter 05 Limits Comparison Chord */
    var limitsGroup = null, limitsChord = null;

    /* Floating DOM Labels */
    var domWordLabels = [];
    var domProbeLabel = null;
    var domTempLabels = [];
    var projVec = null;

    function resetProbeTrail() {
      probeTrailCount = 0;
      if (probeTrailLine) probeTrailLine.geometry.setDrawRange(0, 0);
    }

    function focusWordInCamera(wordId) {
      if (!orbit || !THREE) return;
      var w = WORD_BY_ID[wordId];
      if (!w) return;
      var p = project8to3(w.v, (activeMode() === 'limits' ? state.sliceDeg : 0) * Math.PI / 180);
      orbit.goTo({ target: [p[0] * 0.55, p[1] * 0.55, p[2] * 0.55], dist: 310 });
    }

    function focusAnalogyCamera() {
      if (!orbit) return;
      var an = ANALOGIES.find(function (x) { return x.id === state.analogyId; }) || ANALOGIES[0];
      var wa = WORD_BY_ID[an.a], wt = WORD_BY_ID[an.expected];
      var pa = project8to3(wa.v, 0), pt = project8to3(wt.v, 0);
      var mx = (pa[0] + pt[0]) * 0.45, my = (pa[1] + pt[1]) * 0.45, mz = (pa[2] + pt[2]) * 0.45;
      var theta = Math.atan2(mx, mz) + 0.35;
      orbit.goTo({ target: [mx, my, mz], dist: 345, phi: 1.16, theta: theta });
    }

    /* Smooth spherical camera orbit rig with smart click-vs-drag detection */
    function createOrbitRig() {
      var g = { target: new THREE.Vector3(0, 0, 0), dist: 455, phi: 1.24, theta: 0.42 };
      var c = { target: g.target.clone(), dist: g.dist, phi: g.phi, theta: g.theta };
      var dragging = false, downX = 0, downY = 0, lastX = 0, lastY = 0, movedDist = 0, pinch = 0;

      function goTo(cfg, instant) {
        if (cfg.target) g.target.fromArray(cfg.target);
        if (cfg.dist != null) g.dist = Math.max(150, Math.min(820, cfg.dist));
        if (cfg.phi != null) g.phi = Math.max(0.22, Math.min(Math.PI - 0.22, cfg.phi));
        if (cfg.theta != null) g.theta = cfg.theta;
        if (instant) { c.target.copy(g.target); c.dist = g.dist; c.phi = g.phi; c.theta = g.theta; }
      }

      function dismissHint() { if (hintEl) hintEl.classList.add('is-gone'); }
      setTimeout(dismissHint, 7500);

      canvas.addEventListener('pointerdown', function (e) {
        dragging = true; movedDist = 0;
        downX = lastX = e.clientX; downY = lastY = e.clientY;
        canvas.setPointerCapture && canvas.setPointerCapture(e.pointerId);
        dismissHint();
      });

      canvas.addEventListener('pointermove', function (e) {
        if (dragging) {
          var dx = e.clientX - lastX, dy = e.clientY - lastY;
          movedDist += Math.hypot(dx, dy);
          if (movedDist > 5) canvas.classList.add('is-dragging');
          g.theta -= dx * 0.0058;
          g.phi = Math.max(0.22, Math.min(Math.PI - 0.22, g.phi - dy * 0.0046));
          lastX = e.clientX; lastY = e.clientY;
          return;
        }
        /* Hover raycasting for 3D words */
        performHoverTest(e.clientX, e.clientY);
      });

      function onUp(e) {
        if (dragging && movedDist < 7 && e) {
          performClickTest(e.clientX, e.clientY);
        }
        dragging = false;
        canvas.classList.remove('is-dragging');
        if (e && e.pointerId != null && canvas.releasePointerCapture) {
          try { canvas.releasePointerCapture(e.pointerId); } catch (_) {}
        }
      }
      canvas.addEventListener('pointerup', onUp);
      canvas.addEventListener('pointercancel', onUp);
      canvas.addEventListener('pointerleave', function () {
        dragging = false;
        canvas.classList.remove('is-dragging');
        canvas.classList.remove('is-hovering');
        state.hoverWordId = null;
      });

      canvas.addEventListener('wheel', function (e) {
        e.preventDefault();
        g.dist = Math.max(150, Math.min(820, g.dist * (1 + (e.deltaY > 0 ? 0.11 : -0.11))));
        dismissHint();
      }, { passive: false });

      canvas.addEventListener('touchmove', function (e) {
        if (e.touches.length !== 2) return;
        e.preventDefault();
        var d = Math.hypot(e.touches[0].clientX - e.touches[1].clientX,
                           e.touches[0].clientY - e.touches[1].clientY);
        if (pinch) g.dist = Math.max(150, Math.min(820, g.dist * (pinch / d)));
        pinch = d;
      }, { passive: false });
      canvas.addEventListener('touchend', function () { pinch = 0; });

      return {
        goTo: goTo,
        update: function (dt) {
          if (state.autoOrbit && !dragging) {
            g.theta += dt * (activeMode() === 'temperature' ? 0.08 : 0.11);
          }
          var aspect = (camera && camera.aspect) || 1.6;
          var responsiveBoost = aspect < 1.05 ? (1.05 / Math.max(0.65, aspect)) : 1.0;
          var goalDist = Math.min(820, g.dist * responsiveBoost);

          var k = 1 - Math.pow(0.0012, Math.min(dt || 0.016, 0.05));
          c.dist += (goalDist - c.dist) * k;
          c.phi += (g.phi - c.phi) * k;
          c.theta += (g.theta - c.theta) * k;
          c.target.lerp(g.target, k);

          var sp = Math.sin(c.phi);
          camera.position.set(
            c.target.x + c.dist * sp * Math.sin(c.theta),
            c.target.y + c.dist * Math.cos(c.phi),
            c.target.z + c.dist * sp * Math.cos(c.theta)
          );
          camera.lookAt(c.target);
        }
      };
    }

    function screenToNDC(clientX, clientY) {
      var r = canvas.getBoundingClientRect();
      return {
        x: ((clientX - r.left) / r.width) * 2 - 1,
        y: -(((clientY - r.top) / r.height) * 2 - 1)
      };
    }

    function performHoverTest(cx, cy) {
      if (!raycaster || !camera || activeMode() === 'temperature') {
        state.hoverWordId = null;
        canvas.classList.remove('is-hovering');
        return;
      }
      var ndc = screenToNDC(cx, cy);
      raycaster.setFromCamera(ndc, camera);
      var hits = raycaster.intersectObjects(wordHitTargets, false);
      if (hits.length) {
        state.hoverWordId = hits[0].object.userData.wordId;
        canvas.classList.add('is-hovering');
      } else {
        state.hoverWordId = null;
        canvas.classList.remove('is-hovering');
      }
    }

    function performClickTest(cx, cy) {
      if (!raycaster || !camera) return;
      var m = activeMode();
      var ndc = screenToNDC(cx, cy);
      raycaster.setFromCamera(ndc, camera);

      if (m === 'temperature') {
        /* Clicking a pillar selects/inspects it or triggers a live sample */
        var pHits = raycaster.intersectObjects(tempPillars.map(function (p) { return p.mesh; }), false);
        if (pHits.length) {
          state.sampledIdx = pHits[0].object.userData.candIdx;
          state.sampleFlash = 1.0;
          paintTempPanel();
        }
        return;
      }

      var hits = raycaster.intersectObjects(wordHitTargets, false);
      if (hits.length) {
        var wid = hits[0].object.userData.wordId;
        openInspector(wid);
      }
    }

    /* Helper: Create a sleek 3D vector arrow (shaft + cone tip) */
    function makeArrow3D(colorHex) {
      var grp = new THREE.Group();
      var shaftMat = new THREE.MeshBasicMaterial({ color: colorHex, transparent: true, opacity: 0.92 });
      var shaft = new THREE.Mesh(new THREE.CylinderGeometry(1.5, 1.5, 1, 12), shaftMat);
      shaft.position.y = 0.5;
      var tip = new THREE.Mesh(new THREE.ConeGeometry(4.6, 13, 16), shaftMat);
      grp.add(shaft, tip);
      grp.userData = { shaft: shaft, tip: tip };
      return grp;
    }

    function orientArrow3D(arrowGrp, fromVec, toVec) {
      var dir = new THREE.Vector3().subVectors(toVec, fromVec);
      var len = dir.length();
      if (len < 1) { arrowGrp.visible = false; return; }
      arrowGrp.visible = true;
      arrowGrp.position.copy(fromVec);
      var up = new THREE.Vector3(0, 1, 0);
      arrowGrp.quaternion.setFromUnitVectors(up, dir.clone().normalize());
      var tipLen = Math.min(13, len * 0.35);
      var shaftLen = Math.max(0.5, len - tipLen);
      arrowGrp.userData.shaft.scale.set(1, shaftLen, 1);
      arrowGrp.userData.shaft.position.y = shaftLen * 0.5;
      arrowGrp.userData.tip.position.y = shaftLen + tipLen * 0.45;
    }

    /* Build entire Three.js scene once Three loads */
    function buildScene() {
      scene = new THREE.Scene();
      scene.fog = new THREE.FogExp2(0x05070B, 0.00088);

      var rect = canvas.getBoundingClientRect();
      camera = new THREE.PerspectiveCamera(44, Math.max(1, rect.width) / Math.max(1, rect.height), 1, 3500);
      raycaster = new THREE.Raycaster();
      projVec = new THREE.Vector3();

      /* Studio-grade 3D lighting for faceted crystals & holographic pillars */
      scene.add(new THREE.AmbientLight(0x7DA8E8, 0.72));
      var keyLight = new THREE.PointLight(0x8FDBFF, 1.5, 1100);
      keyLight.position.set(220, 260, 240);
      scene.add(keyLight);
      var rimLight = new THREE.PointLight(0xE31B23, 1.15, 950);
      rimLight.position.set(-240, -140, -210);
      scene.add(rimLight);

      var glowTex = makeGlowTexture(THREE, 'rgba(255,255,255,0.95)', 'rgba(255,255,255,0.34)');
      var softCloudTex = makeGlowTexture(THREE, 'rgba(255,255,255,0.22)', 'rgba(255,255,255,0.06)');

      /* ---- Deep Space Ambient Dust Cloud ---- */
      var dustGeo = new THREE.BufferGeometry();
      var dustPos = new Float32Array(420 * 3);
      for (var d = 0; d < 420; d++) {
        var r = 240 + Math.random() * 520;
        var th = Math.random() * Math.PI * 2;
        var ph = Math.acos(2 * Math.random() - 1);
        dustPos[d * 3]     = r * Math.sin(ph) * Math.cos(th);
        dustPos[d * 3 + 1] = r * Math.cos(ph);
        dustPos[d * 3 + 2] = r * Math.sin(ph) * Math.sin(th);
      }
      dustGeo.setAttribute('position', new THREE.BufferAttribute(dustPos, 3));
      dustPoints = new THREE.Points(dustGeo, new THREE.PointsMaterial({
        color: 0x8FDBFF, size: 2.2, transparent: true, opacity: 0.34,
        blending: THREE.AdditiveBlending, depthWrite: false
      }));
      scene.add(dustPoints);

      /* ---- Semantic Galaxy Group ---- */
      galaxyGroup = new THREE.Group();
      scene.add(galaxyGroup);

      /* Translucent Geodesic Unit Hypersphere Shell + Equatorial Rings */
      sphereShell = new THREE.Group();
      var icoGeo = new THREE.IcosahedronGeometry(RADIUS * 1.07, 2);
      var wireGeo = new THREE.WireframeGeometry(icoGeo);
      var wireMat = new THREE.LineBasicMaterial({ color: 0x2B6CFF, transparent: true, opacity: 0.085 });
      sphereShell.add(new THREE.LineSegments(wireGeo, wireMat));

      var eqRing = new THREE.Mesh(
        new THREE.TorusGeometry(RADIUS * 1.07, 0.55, 8, 96),
        new THREE.MeshBasicMaterial({ color: 0x8FDBFF, transparent: true, opacity: 0.14 })
      );
      eqRing.rotation.x = Math.PI / 2;
      sphereShell.add(eqRing);

      var merRing = new THREE.Mesh(
        new THREE.TorusGeometry(RADIUS * 1.07, 0.45, 8, 96),
        new THREE.MeshBasicMaterial({ color: 0x2B6CFF, transparent: true, opacity: 0.11 })
      );
      sphereShell.add(merRing);

      /* Subtle 3D Cartesian axes inside the hypersphere */
      var axGeo = new THREE.BufferGeometry();
      var axL = RADIUS * 1.15;
      axGeo.setAttribute('position', new THREE.Float32BufferAttribute([
        -axL, 0, 0, axL, 0, 0,
        0, -axL, 0, 0, axL, 0,
        0, 0, -axL, 0, 0, axL
      ], 3));
      sphereShell.add(new THREE.LineSegments(axGeo, new THREE.LineBasicMaterial({
        color: 0x8FDBFF, transparent: true, opacity: 0.12
      })));
      galaxyGroup.add(sphereShell);

      /* Cluster Ambient Volumetric Halos */
      Object.keys(CLUSTERS).forEach(function (ck) {
        var members = WORDS.filter(function (w) { return w.cluster === ck; });
        var cx = 0, cy = 0, cz = 0;
        members.forEach(function (w) {
          var p = project8to3(w.v, 0);
          cx += p[0]; cy += p[1]; cz += p[2];
        });
        cx /= members.length; cy /= members.length; cz /= members.length;
        var sp = new THREE.Sprite(new THREE.SpriteMaterial({
          map: softCloudTex, color: CLUSTERS[ck].color,
          transparent: true, opacity: 0.45, blending: THREE.AdditiveBlending, depthWrite: false
        }));
        sp.position.set(cx, cy, cz);
        sp.scale.set(165, 165, 1);
        sp.userData.clusterKey = ck;
        galaxyGroup.add(sp);
      });

      /* 36 Word Crystals + Glow Auras + Invisible Larger Raycast Hit Spheres */
      var crystalGeo = new THREE.IcosahedronGeometry(5.1, 1);
      var hitGeo = new THREE.SphereGeometry(13.5, 10, 10);
      var hitMat = new THREE.MeshBasicMaterial({ visible: false });

      WORDS.forEach(function (w, i) {
        var p = project8to3(w.v, 0);
        var pos = new THREE.Vector3(p[0], p[1], p[2]);
        wordPositions.push(pos);

        var mat = new THREE.MeshStandardMaterial({
          color: w.colorInt,
          emissive: w.colorInt,
          emissiveIntensity: 0.52,
          roughness: 0.25,
          metalness: 0.35
        });
        var mesh = new THREE.Mesh(crystalGeo, mat);
        mesh.position.copy(pos);
        mesh.userData.wordId = w.id;
        galaxyGroup.add(mesh);
        wordMeshes.push(mesh);

        var aura = new THREE.Sprite(new THREE.SpriteMaterial({
          map: glowTex, color: w.colorInt, transparent: true, opacity: 0.56,
          blending: THREE.AdditiveBlending, depthWrite: false
        }));
        aura.position.copy(pos);
        aura.scale.set(28, 28, 1);
        galaxyGroup.add(aura);
        wordAuras.push(aura);

        var hit = new THREE.Mesh(hitGeo, hitMat);
        hit.position.copy(pos);
        hit.userData.wordId = w.id;
        galaxyGroup.add(hit);
        wordHitTargets.push(hit);

        /* DOM Label */
        var lbl = document.createElement('span');
        lbl.className = 'gmx-label';
        lbl.style.setProperty('--c', w.colorHex);
        lbl.innerHTML = '<i></i><b>' + w.fa + '</b>';
        labelLayer.appendChild(lbl);
        domWordLabels.push(lbl);
      });

      /* Semantic Proximity Constellation Filaments (cos > 0.68 in real 8D!) */
      var filPairs = [];
      for (var a = 0; a < WORDS.length; a++) {
        for (var b = a + 1; b < WORDS.length; b++) {
          var sim = cosSim(WORDS[a].v, WORDS[b].v);
          if (sim > 0.68) filPairs.push([a, b, sim]);
        }
      }
      var filGeo = new THREE.BufferGeometry();
      var filPos = new Float32Array(filPairs.length * 6);
      filGeo.setAttribute('position', new THREE.BufferAttribute(filPos, 3));
      filamentLines = new THREE.LineSegments(filGeo, new THREE.LineBasicMaterial({
        color: 0x8FDBFF, transparent: true, opacity: 0.17, blending: THREE.AdditiveBlending
      }));
      filamentLines.userData.pairs = filPairs;
      galaxyGroup.add(filamentLines);

      /* Dynamic Hover / Selected Neighbor Beams (top 5 real 8D neighbors) */
      var nbGeo = new THREE.BufferGeometry();
      nbGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(5 * 6), 3));
      neighborLines = new THREE.LineSegments(nbGeo, new THREE.LineBasicMaterial({
        color: 0x8FDBFF, transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending
      }));
      neighborLines.visible = false;
      galaxyGroup.add(neighborLines);

      /* ---- Chapter 02: Analogy Parallelogram 3D Group ---- */
      algebraGroup = new THREE.Group();
      algebraGroup.visible = false;
      arrowSub = makeArrow3D(0xFF5A5A);
      arrowAdd = makeArrow3D(0x8FDBFF);
      algebraGroup.add(arrowSub, arrowAdd);

      var parGeo = new THREE.BufferGeometry();
      parGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
      dashedParallel = new THREE.Line(parGeo, new THREE.LineDashedMaterial({
        color: 0xF5B754, dashSize: 6, gapSize: 4, transparent: true, opacity: 0.65
      }));
      algebraGroup.add(dashedParallel);

      ghostOrb = new THREE.Mesh(
        new THREE.OctahedronGeometry(7.2, 0),
        new THREE.MeshStandardMaterial({ color: 0xFFFFFF, emissive: 0x8FDBFF, emissiveIntensity: 0.9, roughness: 0.2 })
      );
      algebraGroup.add(ghostOrb);

      targetRing = new THREE.Mesh(
        new THREE.TorusGeometry(15, 1.1, 10, 48),
        new THREE.MeshBasicMaterial({ color: 0x3EE0A8, transparent: true, opacity: 0.85 })
      );
      algebraGroup.add(targetRing);
      galaxyGroup.add(algebraGroup);

      /* ---- Chapter 03: Polysemous "شیر" Crystal & Attention Beams ---- */
      attentionGroup = new THREE.Group();
      attentionGroup.visible = false;

      var initProbe = project8to3(PROBE_STATIC, 0);
      probeCurrentPos = new THREE.Vector3(initProbe[0], initProbe[1], initProbe[2]);

      probeMesh = new THREE.Mesh(
        new THREE.OctahedronGeometry(9.2, 0),
        new THREE.MeshStandardMaterial({
          color: 0xFF3A44, emissive: 0xE31B23, emissiveIntensity: 0.88,
          roughness: 0.18, metalness: 0.4
        })
      );
      probeMesh.position.copy(probeCurrentPos);
      attentionGroup.add(probeMesh);

      probeRing1 = new THREE.Mesh(
        new THREE.TorusGeometry(15.5, 0.7, 8, 48),
        new THREE.MeshBasicMaterial({ color: 0xFF6A5A, transparent: true, opacity: 0.72 })
      );
      probeRing2 = new THREE.Mesh(
        new THREE.TorusGeometry(20.5, 0.45, 8, 48),
        new THREE.MeshBasicMaterial({ color: 0xF5B754, transparent: true, opacity: 0.45 })
      );
      attentionGroup.add(probeRing1, probeRing2);

      probeAura = new THREE.Sprite(new THREE.SpriteMaterial({
        map: glowTex, color: 0xE31B23, transparent: true, opacity: 0.85,
        blending: THREE.AdditiveBlending, depthWrite: false
      }));
      probeAura.scale.set(54, 54, 1);
      attentionGroup.add(probeAura);

      /* Up to 6 curved Attention Bezier Arcs + 3 photon pulses per curve */
      for (var c = 0; c < 6; c++) {
        var lg = new THREE.BufferGeometry();
        lg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(32 * 3), 3));
        var lm = new THREE.Line(lg, new THREE.LineBasicMaterial({
          color: 0xF5B754, transparent: true, opacity: 0.75, blending: THREE.AdditiveBlending
        }));
        lm.visible = false;
        attentionGroup.add(lm);
        attLineMeshes.push(lm);

        var pulses = [];
        for (var pIdx = 0; pIdx < 5; pIdx++) {
          var pm = new THREE.Sprite(new THREE.SpriteMaterial({
            map: glowTex, color: 0xFFFFFF, transparent: true, opacity: 0.95,
            blending: THREE.AdditiveBlending, depthWrite: false
          }));
          pm.scale.set(18, 18, 1);
          pm.visible = false;
          attentionGroup.add(pm);
          pulses.push(pm);
        }
        attPulseMeshes.push(pulses);
      }

      /* 3D Residual Trajectory Trail of "شیر" */
      var MAX_TRAIL = 140;
      probeTrailArr = new Float32Array(MAX_TRAIL * 3);
      var trGeo = new THREE.BufferGeometry();
      trGeo.setAttribute('position', new THREE.BufferAttribute(probeTrailArr, 3));
      trGeo.setDrawRange(0, 0);
      probeTrailLine = new THREE.Line(trGeo, new THREE.LineBasicMaterial({
        color: 0xF5B754, transparent: true, opacity: 0.72, blending: THREE.AdditiveBlending
      }));
      probeTrailLine.frustumCulled = false;
      attentionGroup.add(probeTrailLine);
      galaxyGroup.add(attentionGroup);

      /* Probe DOM label */
      domProbeLabel = document.createElement('span');
      domProbeLabel.className = 'gmx-label is-probe';
      domProbeLabel.style.setProperty('--c', '#FF4D55');
      domProbeLabel.innerHTML = '<i></i><b>شیر</b><em>Query</em>';
      labelLayer.appendChild(domProbeLabel);

      /* ---- Chapter 04: 3D Holographic Temperature / Softmax Arena ---- */
      tempGroup = new THREE.Group();
      tempGroup.visible = false;
      scene.add(tempGroup);

      /* Arena floor radar rings */
      [58, 112, 162].forEach(function (rad, idx) {
        var rMesh = new THREE.Mesh(
          new THREE.RingGeometry(rad - 0.8, rad, 72),
          new THREE.MeshBasicMaterial({
            color: idx === 1 ? 0x8FDBFF : 0x2B6CFF,
            transparent: true, opacity: idx === 1 ? 0.25 : 0.14, side: THREE.DoubleSide
          })
        );
        rMesh.rotation.x = -Math.PI / 2;
        rMesh.position.y = -48;
        tempGroup.add(rMesh);
      });

      /* Central Prompt Crystal */
      tempCore = new THREE.Mesh(
        new THREE.OctahedronGeometry(14, 0),
        new THREE.MeshStandardMaterial({
          color: 0x8FDBFF, emissive: 0x2B6CFF, emissiveIntensity: 0.85,
          roughness: 0.2, metalness: 0.45
        })
      );
      tempCore.position.set(0, -18, 0);
      tempGroup.add(tempCore);

      tempCoreRing = new THREE.Mesh(
        new THREE.TorusGeometry(24, 0.8, 8, 48),
        new THREE.MeshBasicMaterial({ color: 0x8FDBFF, transparent: true, opacity: 0.55 })
      );
      tempCoreRing.position.set(0, -18, 0);
      tempCoreRing.rotation.x = Math.PI / 2;
      tempGroup.add(tempCoreRing);

      /* 10 Radial 3D Probability Pillars */
      var pRadius = 116;
      for (var pi = 0; pi < 10; pi++) {
        var ang = (pi / 10) * Math.PI * 2 - Math.PI / 2;
        var px = Math.cos(ang) * pRadius;
        var pz = Math.sin(ang) * pRadius;

        var pilMat = new THREE.MeshStandardMaterial({
          color: 0x3EE0A8, emissive: 0x3EE0A8, emissiveIntensity: 0.55,
          transparent: true, opacity: 0.86, roughness: 0.22, metalness: 0.25
        });
        var pilMesh = new THREE.Mesh(new THREE.CylinderGeometry(8.2, 9.6, 1, 20), pilMat);
        pilMesh.position.set(px, -48, pz);
        pilMesh.userData.candIdx = pi;
        tempGroup.add(pilMesh);

        var capAura = new THREE.Sprite(new THREE.SpriteMaterial({
          map: glowTex, color: 0x3EE0A8, transparent: true, opacity: 0.7,
          blending: THREE.AdditiveBlending, depthWrite: false
        }));
        capAura.scale.set(34, 34, 1);
        tempGroup.add(capAura);

        var baseRing = new THREE.Mesh(
          new THREE.RingGeometry(10.5, 12.2, 24),
          new THREE.MeshBasicMaterial({ color: 0x8FDBFF, transparent: true, opacity: 0.35, side: THREE.DoubleSide })
        );
        baseRing.rotation.x = -Math.PI / 2;
        baseRing.position.set(px, -47.6, pz);
        tempGroup.add(baseRing);

        tempPillars.push({
          idx: pi, x: px, z: pz, mesh: pilMesh, mat: pilMat,
          aura: capAura, baseRing: baseRing, curH: 8, targetH: 8
        });

        var tLbl = document.createElement('span');
        tLbl.className = 'gmx-label';
        tLbl.innerHTML = '<i></i><b></b><em></em>';
        labelLayer.appendChild(tLbl);
        domTempLabels.push(tLbl);
      }

      /* Top-P Nucleus Horizon Ring */
      tempTopPRing = new THREE.Mesh(
        new THREE.TorusGeometry(pRadius, 0.9, 8, 72),
        new THREE.MeshBasicMaterial({ color: 0xF5B754, transparent: true, opacity: 0.45 })
      );
      tempTopPRing.rotation.x = Math.PI / 2;
      tempTopPRing.position.y = -32;
      tempGroup.add(tempTopPRing);

      /* Sampled Volumetric 3D Energy Laser Beam + Shockwave Ring */
      tempLaserLine = makeArrow3D(0xF5B754);
      tempGroup.add(tempLaserLine);

      tempShockRing = new THREE.Mesh(
        new THREE.RingGeometry(12, 15.5, 36),
        new THREE.MeshBasicMaterial({ color: 0xF5B754, transparent: true, opacity: 0, side: THREE.DoubleSide })
      );
      tempShockRing.rotation.x = -Math.PI / 2;
      tempGroup.add(tempShockRing);

      /* ---- Chapter 05: 8D Shadow Comparison Chord ---- */
      limitsGroup = new THREE.Group();
      limitsGroup.visible = false;
      var limGeo = new THREE.BufferGeometry();
      limGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
      limitsChord = new THREE.Line(limGeo, new THREE.LineDashedMaterial({
        color: 0xFF6A5A, dashSize: 7, gapSize: 5, transparent: true, opacity: 0.85
      }));
      limitsGroup.add(limitsChord);
      galaxyGroup.add(limitsGroup);
    }

    /* =====================================================================
       7. PER-FRAME UPDATE & DOM LABEL COLLISION RESOLVER
       ===================================================================== */
    function updateScene(dt, t) {
      var m = activeMode();
      var inTemp = (m === 'temperature');

      /* Subtle dust & shell motion */
      if (dustPoints) dustPoints.rotation.y = t * 0.015;
      if (sphereShell) {
        sphereShell.rotation.y = t * 0.012;
        sphereShell.visible = !inTemp;
      }

      /* Toggle primary 3D arenas */
      if (galaxyGroup) galaxyGroup.visible = !inTemp;
      if (tempGroup)   tempGroup.visible   = inTemp;

      if (!inTemp) {
        var phiRad = (m === 'limits') ? (state.sliceDeg * Math.PI / 180) : 0;

        /* Update all 36 word coordinates from exact 8D->3D projection */
        for (var i = 0; i < WORDS.length; i++) {
          var p = project8to3(WORDS[i].v, phiRad);
          var tgt = wordPositions[i];
          tgt.x += (p[0] - tgt.x) * Math.min(1, dt * 8 + 0.08);
          tgt.y += (p[1] - tgt.y) * Math.min(1, dt * 8 + 0.08);
          tgt.z += (p[2] - tgt.z) * Math.min(1, dt * 8 + 0.08);

          var bob = Math.sin(t * 1.15 + i * 0.7) * 1.4;
          wordMeshes[i].position.set(tgt.x, tgt.y + bob, tgt.z);
          wordMeshes[i].rotation.y = t * 0.55 + i;
          wordMeshes[i].rotation.x = t * 0.35 + i * 0.4;
          wordAuras[i].position.copy(wordMeshes[i].position);
          wordHitTargets[i].position.copy(wordMeshes[i].position);

          var isSel = (WORDS[i].id === state.selectedWordId);
          var isHov = (WORDS[i].id === state.hoverWordId);
          var sc = isSel ? 1.65 : (isHov ? 1.4 : 1.0);
          wordMeshes[i].scale.setScalar(sc);
          wordAuras[i].scale.setScalar(isSel ? 52 : (isHov ? 42 : 27));
        }

        /* Update proximity filaments */
        var pairs = filamentLines.userData.pairs;
        var fArr = filamentLines.geometry.attributes.position.array;
        for (var fp = 0; fp < pairs.length; fp++) {
          var pa = wordMeshes[pairs[fp][0]].position;
          var pb = wordMeshes[pairs[fp][1]].position;
          var o = fp * 6;
          fArr[o] = pa.x; fArr[o + 1] = pa.y; fArr[o + 2] = pa.z;
          fArr[o + 3] = pb.x; fArr[o + 4] = pb.y; fArr[o + 5] = pb.z;
        }
        filamentLines.geometry.attributes.position.needsUpdate = true;

        /* Update hover/selected top-5 neighbor beams */
        var focalId = state.hoverWordId || state.selectedWordId;
        if (focalId && WORD_BY_ID[focalId]) {
          var fw = WORD_BY_ID[focalId];
          var fPos = wordMeshes[fw.idx].position;
          var top5 = WORDS.filter(function (o) { return o.id !== fw.id; }).map(function (o) {
            return { w: o, s: cosSim(fw.v, o.v) };
          }).sort(function (a, b) { return b.s - a.s; }).slice(0, 5);

          var nArr = neighborLines.geometry.attributes.position.array;
          for (var ni = 0; ni < 5; ni++) {
            var np = wordMeshes[top5[ni].w.idx].position;
            var no = ni * 6;
            nArr[no] = fPos.x; nArr[no + 1] = fPos.y; nArr[no + 2] = fPos.z;
            nArr[no + 3] = np.x; nArr[no + 4] = np.y; nArr[no + 5] = np.z;
          }
          neighborLines.geometry.attributes.position.needsUpdate = true;
          neighborLines.material.color.setHex(fw.colorInt);
          neighborLines.visible = true;
        } else {
          neighborLines.visible = false;
        }

        /* ---- Chapter 02: Analogy 3D Parallelogram ---- */
        algebraGroup.visible = (m === 'algebra');
        if (algebraGroup.visible) {
          var an = ANALOGIES.find(function (x) { return x.id === state.analogyId; }) || ANALOGIES[0];
          var wa = WORD_BY_ID[an.a], wb = WORD_BY_ID[an.b], wc = WORD_BY_ID[an.c], wt = WORD_BY_ID[an.expected];
          var posA = wordMeshes[wa.idx].position;
          var posB = wordMeshes[wb.idx].position;
          var posC = wordMeshes[wc.idx].position;
          var posT = wordMeshes[wt.idx].position;

          /* Exact 3D Parallelogram: A + (C - B) = parallel transport of (B -> C) onto A! */
          var deltaBC = new THREE.Vector3().subVectors(posC, posB);
          var exactTarget = new THREE.Vector3().copy(posA).add(deltaBC);
          var prog = state.analogyProgress;

          /* Gold/Crimson base vector: B -> C (the semantic shift vector Delta) */
          var pBase = Math.min(1, prog * 1.4 + 0.35);
          var tipBC = new THREE.Vector3().lerpVectors(posB, posC, pBase);
          orientArrow3D(arrowSub, posB, tipBC);

          /* Ion-Cyan transported vector: A -> A + prog * (C - B) landing on target */
          var tipA = new THREE.Vector3().lerpVectors(posA, exactTarget, prog);
          if (prog > 0.02) orientArrow3D(arrowAdd, posA, tipA);
          else arrowAdd.visible = false;

          /* Dashed parallelogram edge connecting B -> A so the parallel shift is unmistakable */
          var pArr = dashedParallel.geometry.attributes.position.array;
          pArr[0] = posB.x; pArr[1] = posB.y; pArr[2] = posB.z;
          pArr[3] = posA.x; pArr[4] = posA.y; pArr[5] = posA.z;
          dashedParallel.geometry.attributes.position.needsUpdate = true;
          dashedParallel.computeLineDistances();

          ghostOrb.position.copy(tipA);
          ghostOrb.rotation.y = t * 1.8;
          ghostOrb.rotation.x = t * 1.2;

          targetRing.position.copy(posT);
          targetRing.lookAt(camera.position);
          var pulse = 1 + Math.sin(t * 3.5) * 0.14;
          targetRing.scale.setScalar(pulse);
        }

        /* ---- Chapter 03: Attention & "شیر" ---- */
        attentionGroup.visible = (m === 'attention');
        if (attentionGroup.visible) {
          var att = computeAttentionState(state.sentenceId, state.layer);
          var prj = project8to3(att.vec, 0);
          var goalPos = new THREE.Vector3(prj[0], prj[1], prj[2]);
          probeCurrentPos.lerp(goalPos, Math.min(1, dt * 6.5 + 0.06));

          probeMesh.position.copy(probeCurrentPos);
          probeMesh.rotation.y = t * 1.1;
          probeMesh.rotation.z = t * 0.7;
          probeMesh.material.emissive.setHex(att.sentence.color);
          probeAura.position.copy(probeCurrentPos);
          probeAura.material.color.setHex(att.sentence.color);

          probeRing1.position.copy(probeCurrentPos);
          probeRing1.rotation.x = t * 0.9;
          probeRing1.rotation.y = t * 0.6;
          probeRing1.material.color.setHex(att.sentence.color);

          probeRing2.position.copy(probeCurrentPos);
          probeRing2.rotation.x = -t * 0.5 + 0.8;
          probeRing2.rotation.z = t * 0.8;

          /* Record smooth 3D trajectory trail */
          var MAX_T = 140;
          if (probeTrailCount === 0 ||
              Math.hypot(
                probeCurrentPos.x - probeTrailArr[(probeTrailCount - 1) * 3],
                probeCurrentPos.y - probeTrailArr[(probeTrailCount - 1) * 3 + 1],
                probeCurrentPos.z - probeTrailArr[(probeTrailCount - 1) * 3 + 2]
              ) > 2.2) {
            if (probeTrailCount >= MAX_T) {
              probeTrailArr.copyWithin(0, 3);
              probeTrailCount = MAX_T - 1;
            }
            probeTrailArr[probeTrailCount * 3]     = probeCurrentPos.x;
            probeTrailArr[probeTrailCount * 3 + 1] = probeCurrentPos.y;
            probeTrailArr[probeTrailCount * 3 + 2] = probeCurrentPos.z;
            probeTrailCount++;
            probeTrailLine.geometry.attributes.position.needsUpdate = true;
            probeTrailLine.geometry.setDrawRange(0, probeTrailCount);
          }

          /* Draw curved 3D Bezier Attention Beams + luminous energy stream photons */
          for (var ai = 0; ai < 6; ai++) {
            var lineM = attLineMeshes[ai];
            var pSet = attPulseMeshes[ai];
            if (ai < att.sentence.tokens.length) {
              var tokW = WORD_BY_ID[att.sentence.tokens[ai]];
              var startP = wordMeshes[tokW.idx].position;
              var midP = new THREE.Vector3().addVectors(startP, probeCurrentPos).multiplyScalar(0.5);
              midP.y += 42 + ai * 8;
              var curve = new THREE.QuadraticBezierCurve3(startP, midP, probeCurrentPos);
              var pts = curve.getPoints(31);
              var arr = lineM.geometry.attributes.position.array;
              for (var q = 0; q < 32; q++) {
                arr[q * 3] = pts[q].x; arr[q * 3 + 1] = pts[q].y; arr[q * 3 + 2] = pts[q].z;
              }
              lineM.geometry.attributes.position.needsUpdate = true;
              lineM.material.color.setHex(att.sentence.color);
              lineM.material.opacity = 0.52 + att.weights[ai] * 1.4;
              lineM.visible = true;

              for (var k = 0; k < pSet.length; k++) {
                var u = ((t * (0.48 + att.weights[ai] * 0.95) + k / pSet.length + ai * 0.17) % 1);
                var pt = curve.getPoint(u);
                pSet[k].position.copy(pt);
                pSet[k].material.color.setHex(att.sentence.color);
                var psz = 16 + att.weights[ai] * 34;
                pSet[k].scale.set(psz, psz, 1);
                pSet[k].visible = (state.layer > 0.2 || state.sentenceId === 'none');
              }
            } else {
              lineM.visible = false;
              for (var k2 = 0; k2 < pSet.length; k2++) pSet[k2].visible = false;
            }
          }
        }

        /* ---- Chapter 05: 8D Shadow Comparison Chord ---- */
        limitsGroup.visible = (m === 'limits');
        if (limitsGroup.visible) {
          var pk = wordMeshes[WORD_BY_ID.king.idx].position;
          var pp = wordMeshes[WORD_BY_ID.pipe.idx].position;
          var lArr = limitsChord.geometry.attributes.position.array;
          lArr[0] = pk.x; lArr[1] = pk.y; lArr[2] = pk.z;
          lArr[3] = pp.x; lArr[4] = pp.y; lArr[5] = pp.z;
          limitsChord.geometry.attributes.position.needsUpdate = true;
          limitsChord.computeLineDistances();
        }
      } else {
        /* ---- Chapter 04: 3D Holographic Softmax Pillars ---- */
        var sm = computeSoftmax(state.promptId, state.temperature, state.topP);
        tempCore.rotation.y = t * 0.9;
        tempCore.rotation.x = Math.sin(t * 0.7) * 0.35;
        tempCoreRing.rotation.z = t * 0.6;

        if (state.sampleFlash > 0) state.sampleFlash = Math.max(0, state.sampleFlash - dt * 1.4);

        var minActiveH = 160;
        for (var ti = 0; ti < 10; ti++) {
          var item = sm.items[ti];
          var pil = tempPillars[ti];
          pil.targetH = Math.max(6, item.p * 178 + (item.inNucleus ? 6 : 0));
          pil.curH += (pil.targetH - pil.curH) * Math.min(1, dt * 9 + 0.08);
          if (item.inNucleus && pil.curH < minActiveH) minActiveH = pil.curH;

          pil.mesh.scale.set(1, pil.curH, 1);
          pil.mesh.position.y = -48 + pil.curH * 0.5;

          var colHex = !item.inNucleus ? 0x1E293B
                     : item.kind === 'safe' ? 0x3EE0A8
                     : item.kind === 'mid'  ? 0xF5B754 : 0xE31B23;
          var isWin = (ti === state.sampledIdx);
          pil.mat.color.setHex(isWin ? 0xFFF4D6 : colHex);
          pil.mat.emissive.setHex(colHex);
          pil.mat.emissiveIntensity = !item.inNucleus ? 0.12 : (isWin ? 1.1 : 0.55);
          pil.mat.opacity = item.inNucleus ? 0.9 : 0.25;

          pil.aura.position.set(pil.x, -48 + pil.curH + 6, pil.z);
          pil.aura.material.color.setHex(colHex);
          pil.aura.material.opacity = item.inNucleus ? (isWin ? 0.95 : 0.55) : 0.08;
          pil.baseRing.material.color.setHex(isWin ? 0xF5B754 : colHex);
          pil.baseRing.material.opacity = item.inNucleus ? 0.55 : 0.14;
        }

        /* Position Top-P Nucleus Horizon Ring smoothly */
        var ringY = -48 + Math.max(6, minActiveH - 4);
        tempTopPRing.position.y += (ringY - tempTopPRing.position.y) * Math.min(1, dt * 8 + 0.08);
        tempTopPRing.material.opacity = state.topP < 0.99 ? 0.52 : 0.14;

        /* Volumetric 3D laser beam to active sampled candidate + shockwave pulse */
        var winPil = tempPillars[state.sampledIdx] || tempPillars[0];
        orientArrow3D(
          tempLaserLine,
          new THREE.Vector3(0, -18, 0),
          new THREE.Vector3(winPil.x, -48 + winPil.curH, winPil.z)
        );
        tempLaserLine.userData.shaft.material.opacity = 0.58 + state.sampleFlash * 0.42;

        tempShockRing.position.set(winPil.x, -47.2, winPil.z);
        var swScale = 1 + (1 - state.sampleFlash) * 2.6;
        tempShockRing.scale.setScalar(swScale);
        tempShockRing.material.opacity = state.sampleFlash * 0.85;
      }
    }

    /* Project 3D anchors to crisp non-overlapping Persian DOM labels */
    function paintLabels() {
      var r = canvas.getBoundingClientRect();
      var w = r.width, h = r.height;
      if (w < 10 || h < 10) return;
      var m = activeMode();
      var inTemp = (m === 'temperature');

      var candidates = [];

      if (inTemp) {
        for (var j = 0; j < domWordLabels.length; j++) domWordLabels[j].style.opacity = '0';
        domProbeLabel.style.opacity = '0';

        var sm = computeSoftmax(state.promptId, state.temperature, state.topP);
        for (var ti = 0; ti < 10; ti++) {
          var item = sm.items[ti];
          var pil = tempPillars[ti];
          var lbl = domTempLabels[ti];
          projVec.set(pil.x, -48 + pil.curH + 14, pil.z).project(camera);
          if (projVec.z > 1) { lbl.style.opacity = '0'; continue; }
          var lx = (projVec.x * 0.5 + 0.5) * w;
          var ly = (-projVec.y * 0.5 + 0.5) * h;
          var txt = item.word + '|' + (item.inNucleus ? Math.round(item.pEff * 100) : 'cut');
          if (lbl._cache !== txt) {
            lbl._cache = txt;
            lbl.querySelector('b').textContent = item.word;
            lbl.querySelector('em').textContent = item.inNucleus ? (fa(Math.round(item.pEff * 100)) + '٪') : 'حذف';
          }
          var cHex = !item.inNucleus ? '#475569' : (item.kind === 'safe' ? '#3EE0A8' : (item.kind === 'mid' ? '#F5B754' : '#E31B23'));
          lbl.style.setProperty('--c', cHex);
          lbl.classList.toggle('is-highlight', ti === state.sampledIdx);
          lbl.classList.toggle('is-dim', !item.inNucleus);
          candidates.push({ el: lbl, x: lx, y: ly, pri: (ti === state.sampledIdx ? 0 : (item.inNucleus ? 1 : 3)), dim: !item.inNucleus });
        }
      } else {
        for (var k = 0; k < domTempLabels.length; k++) domTempLabels[k].style.opacity = '0';

        /* Highlight sets per mode so key words ALWAYS take priority */
        var priMap = {};
        if (m === 'algebra') {
          var an = ANALOGIES.find(function (x) { return x.id === state.analogyId; }) || ANALOGIES[0];
          priMap[an.a] = 0; priMap[an.b] = 0; priMap[an.c] = 0; priMap[an.expected] = 0;
        } else if (m === 'attention') {
          var s = SENTENCES.find(function (x) { return x.id === state.sentenceId; }) || SENTENCES[0];
          s.tokens.forEach(function (tid) { priMap[tid] = 0; });
          priMap.tiger = 1; priMap.coffee = 1; priMap.pipe = 1;
        } else if (m === 'limits') {
          priMap.king = 0; priMap.pipe = 0; priMap.tehran = 1; priMap.vector = 1;
        }

        if (m === 'attention' && probeCurrentPos) {
          projVec.copy(probeCurrentPos).add(new THREE.Vector3(0, 18, 0)).project(camera);
          if (projVec.z <= 1) {
            candidates.push({
              el: domProbeLabel,
              x: (projVec.x * 0.5 + 0.5) * w,
              y: (-projVec.y * 0.5 + 0.5) * h,
              pri: -1, dim: false
            });
          } else {
            domProbeLabel.style.opacity = '0';
          }
        } else {
          domProbeLabel.style.opacity = '0';
        }

        for (var wi = 0; wi < WORDS.length; wi++) {
          var wd = WORDS[wi];
          var el = domWordLabels[wi];
          projVec.copy(wordMeshes[wi].position).add(new THREE.Vector3(0, 11, 0)).project(camera);
          if (projVec.z > 1) { el.style.opacity = '0'; continue; }
          var isAct = (wd.id === state.selectedWordId || wd.id === state.hoverWordId);
          var isHi = (priMap[wd.id] === 0);
          var pri = isAct ? -0.5 : (priMap[wd.id] !== undefined ? priMap[wd.id] : 2 + projVec.z);
          var isDim = (m === 'algebra' || m === 'attention') && !isHi && !isAct;

          el.classList.toggle('is-active', isAct);
          el.classList.toggle('is-highlight', isHi && !isAct);
          el.classList.toggle('is-dim', isDim);

          candidates.push({
            el: el,
            x: (projVec.x * 0.5 + 0.5) * w,
            y: (-projVec.y * 0.5 + 0.5) * h,
            pri: pri,
            dim: isDim
          });
        }
      }

      /* Priority collision placement with vertical offset ladder so more labels stay legible */
      candidates.sort(function (a, b) { return a.pri - b.pri; });
      var taken = [];
      for (var ci = 0; ci < candidates.length; ci++) {
        var c = candidates[ci];
        var lw = c.el.offsetWidth || 64, lh = c.el.offsetHeight || 22;
        var lx2 = Math.max(lw / 2 + 8, Math.min(w - lw / 2 - 8, c.x));
        var baseY = Math.max(lh / 2 + 52, Math.min(h - lh / 2 - 12, c.y));
        var step = lh + 4;
        var bestY = null;

        for (var sIdx = 0; sIdx < 5 && bestY === null; sIdx++) {
          var candY = baseY + (sIdx === 0 ? 0 : (sIdx % 2 ? -1 : 1) * Math.ceil(sIdx / 2) * step);
          if (candY < lh / 2 + 50 || candY > h - lh / 2 - 10) continue;
          var clash = false;
          for (var tIdx = 0; tIdx < taken.length; tIdx++) {
            var q = taken[tIdx];
            if (Math.abs(lx2 - q.x) < (lw + q.w) * 0.5 + 3 &&
                Math.abs(candY - q.y) < (lh + q.h) * 0.5 + 3) {
              clash = true; break;
            }
          }
          if (!clash) bestY = candY;
        }
        if (bestY === null) {
          c.el.style.opacity = '0';
          continue;
        }
        taken.push({ x: lx2, y: bestY, w: lw, h: lh });
        c.el.style.opacity = c.dim ? '0.45' : '1';
        c.el.style.transform = 'translate(-50%,-50%) translate(' + lx2.toFixed(1) + 'px,' + bestY.toFixed(1) + 'px)';
      }
    }

    function draw(dt) {
      if (!renderer || !scene || !camera) return;
      clock += dt;
      orbit.update(dt);
      updateScene(dt, clock);
      renderer.render(scene, camera);
      paintLabels();
    }

    function frame(ts) {
      raf = requestAnimationFrame(frame);
      var dt = lastTs ? Math.min((ts - lastTs) / 1000, 0.05) : 0.016;
      lastTs = ts;
      if (dt >= 0.049) { if (++slowFrames > 90) { pause(); return; } }
      else if (slowFrames) slowFrames--;
      draw(dt);
    }

    function play() {
      if (!state.ready || state.running || state.dead) return;
      state.running = true; lastTs = 0; slowFrames = 0;
      raf = requestAnimationFrame(frame);
      paintPlayButtons();
    }

    function pause() {
      if (!state.running) return;
      state.running = false;
      cancelAnimationFrame(raf); raf = 0;
      paintPlayButtons();
    }

    var resizeQ = 0;
    function onResize() {
      if (resizeQ) return;
      resizeQ = requestAnimationFrame(function () {
        resizeQ = 0;
        if (!renderer || !camera) return;
        var r = canvas.getBoundingClientRect();
        var rw = Math.max(1, Math.round(r.width));
        var rh = Math.max(1, Math.round(r.height));
        renderer.setSize(rw, rh, false);
        camera.aspect = rw / rh;
        camera.updateProjectionMatrix();
        if (!state.running) draw(0);
      });
    }

    function sayOverlay(html) {
      if (!noteOverlay) return;
      noteOverlay.hidden = false;
      noteOverlay.innerHTML = html;
    }

    function boot() {
      if (state.ready || state.dead) return;
      if (!hasWebGL()) {
        state.dead = true;
        sayOverlay('<b>این آزمایش سه‌بعدی به WebGL نیاز دارد</b>مرورگر شما WebGL را فعال نکرده است؛ کل توضیح علمی، فرمول‌ها و پرسش‌های پرتکرار پایین همین صفحه در دسترس است.');
        return;
      }
      sayOverlay('<b>در حال بارگذاری فضای سه‌بعدی معنا…</b>');
      loadThree().then(function (T) {
        THREE = T;
        var r = canvas.getBoundingClientRect();
        renderer = new THREE.WebGLRenderer({
          canvas: canvas, antialias: true, alpha: true, powerPreference: 'high-performance'
        });
        renderer.setPixelRatio(Math.min(global.devicePixelRatio || 1, 2));
        renderer.setSize(Math.max(1, Math.round(r.width)), Math.max(1, Math.round(r.height)), false);
        if ('outputEncoding' in renderer && THREE.sRGBEncoding) renderer.outputEncoding = THREE.sRGBEncoding;

        buildScene();
        orbit = createOrbitRig();

        state.ready = true;
        if (noteOverlay) noteOverlay.hidden = true;
        syncAllSliders();
        setChapter(state.freeMode ? 'free' : state.chapter, false);
        draw(0);
        if (visible) play();
        else paintPlayButtons();
      }).catch(function (err) {
        state.dead = true;
        console.warn('[navidix/geometry-of-meaning]', err);
        sayOverlay('<b>صحنه‌ی سه‌بعدی بارگذاری نشد</b>توضیح کامل علمی و ریاضی پایین همین صفحه در دسترس است.');
      });
    }

    /* Viewport & Tab Visibility Guards */
    if ('IntersectionObserver' in global) {
      new IntersectionObserver(function (entries) {
        entries.forEach(function (en) {
          visible = en.isIntersecting;
          if (visible) { state.ready ? play() : boot(); }
          else pause();
        });
      }, { rootMargin: '220px 0px' }).observe(stage);
    } else {
      visible = true;
      boot();
    }

    document.addEventListener('visibilitychange', function () {
      if (document.hidden) pause();
      else if (visible && state.ready) play();
    });

    if ('ResizeObserver' in global) new ResizeObserver(onResize).observe(canvas);
    else global.addEventListener('resize', onResize);

    /* Initialise UI text before WebGL even boots */
    syncAllSliders();
    paintAnalogyPanel();
    paintAttentionPanel();
    paintTempPanel();
    updateVisiblePanels();

    var api = {
      openInspector: openInspector,
      closeInspector: closeInspector,
      toggleFullscreen: toggleFullscreen,
      setChapter: setChapter
    };
    global.NVXGeometryLab.api = api;
    return api;
  }

  global.NVXGeometryLab = { mount: mountGeometryLab, api: null };
})(window);
